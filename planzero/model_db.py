import argparse
import datetime
import hashlib
import json
import os
import sqlite3
import uuid
from collections.abc import Iterator

import numpy as np

from .enums import GHG, IPCC_Sector

MODEL_CACHE_ROOT = os.environ['PLANZERO_MODEL_CACHE_ROOT']
assert MODEL_CACHE_ROOT

def adapt_date_iso(val):
    """Adapt datetime.date to ISO 8601 date."""
    return val.isoformat()


def convert_date(val):
    """Convert ISO 8601 date to datetime.date object."""
    return datetime.date.fromisoformat(val.decode())


sqlite3.register_adapter(datetime.date, adapt_date_iso)
sqlite3.register_converter("date", convert_date)

db_filename = "my_database.db"


def connect(timeout=5.0):
    conn = sqlite3.connect(
        db_filename,
        detect_types=sqlite3.PARSE_DECLTYPES, # use register_converter calls above
        timeout=timeout)
    conn.execute("PRAGMA journal_mode = WAL;")
    return conn


def init_db():

    # Connect to a local file (or use ':memory:' for a temporary in-RAM database)
    conn = sqlite3.connect(db_filename)

    conn.execute("PRAGMA journal_mode = WAL;")

    # Samples
    # Indexes the npy data files representing the results of MCMC
    # (family, version) match a record in Models
    # component is a part of the overall model (e.g. "BC, Public Electricity")
    # rand_var is the name of the NumPyro random variable
    # {file_name}.npy will be found within the cache/Samples directory
    #    it defaults to {family}_{version}_{component}_{rand_var}.npy
    #    but if that's too long or contains inconvenient characters
    #    then an alternate string can be used, but the alternate
    #    should still be unique.
    conn.execute("""
    CREATE TABLE IF NOT EXISTS Ndarray (
        component_id TEXT,
        group_id TEXT,
        key TEXT,
        file_name TEXT,
        UNIQUE(component_id, group_id, key)
    );
    """)

    # Models
    # Records correspond roughly to code versions of a modelling approach.
    # family is e.g. "AR2"
    # version is a counter for distinguishing runs
    # version description is like an experiment logbook
    #
    # Sample size is the length of the MCMC chain run for the model
    #
    # data_cutoff is a date-time. The model only models data that were
    #    published before this date. The model is a candidate for making
    #    predictions about data published after this date.
    # 
    conn.execute("""
    CREATE TABLE IF NOT EXISTS Model (
        model_id TEXT PRIMARY KEY,
        family TEXT,
        version REAL,
        version_description TEXT,
        data_cutoff DATE
    );
    """)

    # which model components support which parts of the model
    # The primary key on this table would be something like "a model's NIR element",
    # but that isn't an ID used elsewhere yet.
    # TODO: use integers for these TEXT fields, with some ENUMs, to shrink file a lot
    conn.execute("""
    CREATE TABLE IF NOT EXISTS ComponentMapping (
        model_id TEXT, --TODO: INT
        ghg TEXT, -- TODO: INT
        NIR_sector TEXT, -- TODO: INT
        region TEXT, -- TODO: INT
        component_id TEXT, --TODO: INT
        UNIQUE (model_id, GHG, NIR_sector, region)
    );
    """)

    # Things that all components have
    conn.execute("""
    CREATE TABLE IF NOT EXISTS ComponentType (
        component_id TEXT PRIMARY KEY,
        component_type TEXT, -- "Normal", "BayesianNormal"
        model_id TEXT
    );
    """)

    conn.execute("""
    CREATE TABLE IF NOT EXISTS Component_Normal (
        component_id TEXT PRIMARY KEY,
        location REAL,
        scale REAL,
        v_unit TEXT
    );
    """)

    conn.execute("""
    CREATE TABLE IF NOT EXISTS Component_BayesianNormal (
        component_id TEXT PRIMARY KEY,
        num_samples INTEGER,
        num_warmup INTEGER,
        thinning INTEGER,
        seed INTEGER,
        scale REAL
    );
    """)

    conn.execute("""
    CREATE TABLE IF NOT EXISTS Task (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        parent_id INTEGER,           -- support recursion
        payload TEXT,                -- The JSON or data the worker needs
        payload_dill TEXT,           -- A binary blob that isn't JSON-encodable
        status TEXT DEFAULT 'PENDING',
        worker_id TEXT,              -- To track which process claimed it
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        claimed_at DATETIME,
        completed_at DATETIME
    );
    """)

    # Save changes and close the connection
    conn.commit()
    conn.close()


def component_scopes_by_model_id(model_id):
    with connect() as conn:
        cursor = conn.execute(
        """SELECT NIR_sector, ghg, region, component_id FROM ComponentMapping
        WHERE model_id = ?;""",
        (model_id,))
        for row in cursor:
            yield {
                    'sector': IPCC_Sector(row[0]),
                    'ghg': GHG(row[1]),
                    'region': row[2],
                    'component_id': row[3],
                    }


def component_scope_by_id(component_id):
    with connect() as conn:
        cursor = conn.execute(
        """SELECT NIR_sector, ghg, region FROM ComponentMapping
        WHERE component_id = ?;""",
        (component_id,))
        for row in cursor:
            yield {
                    'sector': IPCC_Sector(row[0]),
                    'ghg': GHG(row[1]),
                    'region': row[2],
                    'component_id': component_id,
                    }


def index_ndarray_group(model_id, component_id, group_id, ndarray_d_keys) -> dict:
    """Create an ndarray group by loading corresponding values from disk
    where they have been cached.

    It's an operation that combines elements of loading and saving. It does
    not modify files on disk.
    """
    rval = {}
    with connect() as conn:
        for key in ndarray_d_keys:
            file_name = f'{component_id}-{key}.npy'
            conn.execute(
                """INSERT INTO Ndarray
                (component_id, group_id, key, file_name)
                VALUES (?, ?, ?, ?);""",
                (component_id, group_id, key, file_name))
            path = os.path.join(MODEL_CACHE_ROOT, model_id, file_name)
            rval[key] = np.load(path, mmap_mode='r')
    return rval


def save_ndarray_group(model_id, component_id, group_id, ndarray_d):
    os.makedirs(os.path.join(MODEL_CACHE_ROOT, model_id), exist_ok=True)
    saved_paths = []
    try:
        with connect() as conn:
            for key, val in ndarray_d.items():
                file_name = f'{component_id}-{key}.npy'
                conn.execute(
                    """INSERT INTO Ndarray
                    (component_id, group_id, key, file_name)
                    VALUES (?, ?, ?, ?);""",
                    (component_id, group_id, key, file_name))
                path = os.path.join(MODEL_CACHE_ROOT, model_id, file_name)
                fp = np.lib.format.open_memmap(
                    path,
                    mode='w+',
                    dtype='float32', # save space
                    shape=val.shape)
                fp[:] = val
                fp.flush()
                saved_paths.append(path)
    except:
        for path in saved_paths:
            os.remove(path)
        raise


def load_ndarray_group(model_id, component_id, group_id):
    rval = {}
    with connect() as conn:
        cursor = conn.execute(
            """
            SELECT key, file_name
            FROM Ndarray
            WHERE component_id = ? AND group_id = ?
            ;""",
            (component_id, group_id,))
        for key, file_name in cursor:
            path = os.path.join(MODEL_CACHE_ROOT, model_id, file_name)
            rval[key] = np.load(path, mmap_mode='r')
    return rval


def params_Normal(component_id):
    with connect() as conn:
        cursor = conn.execute(
            """
            SELECT location, scale, v_unit, data_cutoff, ghg, NIR_sector
            FROM Component_Normal
            JOIN ComponentMapping ON Component_Normal.component_id = ComponentMapping.component_id
            JOIN Model on ComponentMapping.model_id = Model.model_id
            WHERE Component_Normal.component_id = ?
            GROUP BY location, scale, v_unit, data_cutoff, ghg, NIR_sector
            ;""",
        (component_id,))
        n_yielded = 0
        for row in cursor:
            n_yielded += 1
            yield {
                    'location': row[0],
                    'scale': row[1],
                    'v_unit': row[2],
                    'data_cutoff': row[3],
                    'ghg': GHG(row[4]),
                    'sector': IPCC_Sector(row[5]),
                }
        if n_yielded == 0:
            raise NoRecord(component_id)


def params_BayesianNormal(component_id):
    with connect() as conn:
        cursor = conn.execute(
        """SELECT num_samples, num_warmup, thinning, seed, scale, data_cutoff, ghg, NIR_sector, Model.model_id
        FROM Component_BayesianNormal
        JOIN ComponentMapping ON Component_BayesianNormal.component_id = ComponentMapping.component_id
        JOIN Model on ComponentMapping.model_id = Model.model_id
        WHERE Component_BayesianNormal.component_id = ?
        GROUP BY 
                  num_samples, num_warmup, thinning, seed, scale, data_cutoff, ghg, NIR_sector, Model.model_id
            ;""",
        (component_id,))
        n_yielded = 0
        for row in cursor:
            n_yielded += 1
            yield {
                    'num_samples': row[0],
                    'num_warmup': row[1],
                    'thinning': row[2],
                    'seed': row[3],
                    'scale': row[4],
                    'data_cutoff': row[5],
                    'ghg': GHG(row[6]),
                    'sector': IPCC_Sector(row[7]),
                    'model_id': row[8],
                  }
        if n_yielded == 0:
            raise NoRecord(component_id)


def insert_model(cursor, model_id, family, version, data_cutoff,
                 version_description=''):
    query = """INSERT INTO
    Model (model_id, family, version, version_description, data_cutoff)
    VALUES (?, ?, ?, ?, ?);"""
    cursor.execute(
        query,
        (model_id, family, version, version_description, data_cutoff))


def insert_component_mapping(
    cursor,
    model_id,
    ghg,
    NIR_sector,
    region,
    component_id,
    ):
    query = """INSERT INTO
    ComponentMapping (
        model_id, ghg, NIR_sector, region, component_id)
    VALUES (?, ?, ?, ?, ?);"""
    cursor.execute(
        query,
        (model_id, ghg.value, NIR_sector.value, region, component_id))


def insert_component_type(
    cursor,
    component_id,
    component_type,
    model_id,
    ):
    assert component_type in ('Normal', 'BayesianNormal')
    query = """INSERT INTO
    ComponentType (component_id, component_type, model_id)
    VALUES (?, ?, ?);"""
    cursor.execute(
        query,
        (component_id, component_type, model_id))


def insert_component_normal(
    cursor,
    component_id,
    location,
    scale,
    v_unit,
    ):
    query = """INSERT INTO
    Component_Normal (
        component_id, location, scale, v_unit)
    VALUES (?, ?, ?, ?);"""
    cursor.execute(
        query,
        (component_id, location, scale, v_unit))


def insert_component_bayesian_normal(
    cursor,
    component_id,
    num_samples,
    num_warmup,
    thinning,
    seed,
    scale=1,
    ):
    query = """INSERT INTO
    Component_BayesianNormal (
        component_id, num_samples, num_warmup, thinning, seed, scale)
    VALUES (?, ?, ?, ?, ?, ?);"""
    cursor.execute(
        query,
        (component_id, num_samples, num_warmup, thinning, seed, scale))


def insert_task(cursor, payload: dict, parent_id=0):
    """Producer function to add new work to the queue."""
    payload_text = json.dumps(payload)
    cursor.execute(
        "INSERT INTO Task (parent_id, payload) VALUES (?, ?)", 
        (parent_id, payload_text,)
    )


def model_latest_version(family, data_cutoff):
    with connect() as conn:
        cursor = conn.execute(
            """SELECT version, model_id
            FROM Model
            WHERE family = ? AND data_cutoff <= ?
            ORDER BY version DESC
            LIMIT 1;
            """,
            (family, data_cutoff))
        row = cursor.fetchone()
        if row is None:
            raise NoRecord()
        model_version, model_id = row
        return {
                'family': family,
                'data_cutoff': data_cutoff,
                'version': model_version,
                'model_id': model_id,
            }


def components_by_model(model_id):
    with connect() as conn:
        cursor = conn.execute(
            """SELECT component_id, component_type
            FROM ComponentType
            WHERE model_id = ?
            ;""",
            (model_id,))
        for row in cursor:
            yield {
                    'component_id': row[0],
                    'component_type': row[1],
                    }


def normal_components_by_model(model_id):
    with connect() as conn:
        cursor = conn.execute(
            """SELECT ComponentType.component_id, location, scale, v_unit
            FROM ComponentType
            JOIN Component_Normal on ComponentType.component_id = Component_Normal.component_id
            WHERE model_id = ?
            ;""",
            (model_id,))
        for row in cursor:
            yield {
                'component_id': row[0],
                'location': row[1],
                'scale': row[2],
                'v_unit': row[3],
                }


def BayesianNormal_components_by_model(model_id):
    with connect() as conn:
        cursor = conn.execute(
            """SELECT ComponentType.component_id, num_samples, num_warmup, thinning, seed, scale
            FROM ComponentType
            JOIN Component_BayesianNormal on ComponentType.component_id = Component_BayesianNormal.component_id
            WHERE model_id = ?
            ;""",
            (model_id,))
        for row in cursor:
            yield {
                'component_id': row[0],
                'num_samples': row[1],
                'num_warmup': row[2],
                'thinning': row[3],
                'seed': row[4],
                'scale': row[5],
                }


def delete_model(model_id):
    with connect() as conn:
        # TODO: delete the components, the mapping, and the samples
        cursor = conn.execute(
            """DELETE FROM
            Model WHERE model_id = ?
            RETURNING model_id, family, version, data_cutoff;
            """,
            (model_id,))
        returned = list(cursor)
        print('Deleted:', returned)


def main_init_db(args):
    init_db()


def main_task_list(args):

    with connect() as conn:
        cursor = conn.execute('SELECT * from Task ORDER BY created_at ASC')
        for row in cursor:
            print(row)

def main_model_list(args):

    with connect() as conn:
        cursor = conn.execute('SELECT * from Model ORDER BY model_id ASC')
        for row in cursor:
            print(row)

def main_component_type_list(args):

    with connect() as conn:
        cursor = conn.execute(
            '''SELECT * from ComponentType
            ORDER BY component_id ASC''')
        for row in cursor:
            print(row)


def main_model_delete(args):
    for model_id in args.ids:
        delete_model(model_id)


def claim_task(worker_id: str):
    """
    Worker atomically claims the oldest PENDING task.
    Returns a tuple of (task_id, payload_dict) or None if empty.
    """
    with connect() as conn:
        # UPDATE ... RETURNING ensures atomic claim in one step (SQLite 3.35+)
        cursor = conn.execute(
            """
            UPDATE Task
            SET status = 'RUNNING', claimed_at = datetime('now'), worker_id = ?
            WHERE id = (
                SELECT id FROM Task 
                WHERE status = 'PENDING' 
                ORDER BY id ASC LIMIT 1
            )
            RETURNING id, payload;
            """,
            (worker_id,))
        row = cursor.fetchone()
        
        if row:
            return row[0], json.loads(row[1])
        return None, None


class StaleTask(Exception):
    pass


def complete_task(*, task_id: int, worker_id: str):
    """Worker explicitly acknowledges the task is done."""
    with connect() as conn:
        cursor = conn.execute(
            """UPDATE Task
            SET status = 'COMPLETED', completed_at = CURRENT_TIMESTAMP
            WHERE id = ? and worker_id = ?
            RETURNING id
            ; """,
            (task_id, worker_id))
        row = cursor.fetchone()
        if row:
            return
        raise StaleTask()


def main_task_complete(args):
    with connect() as conn:
        cursor = conn.execute(
            """UPDATE Task
            SET status = 'COMPLETED', completed_at = CURRENT_TIMESTAMP
            WHERE id = ?
            RETURNING id
            ; """,
            (args.task_id,))
        row = cursor.fetchone()
        if row:
            return
        raise StaleTask()


def main_task_reset(args):

    with connect() as conn:
        assert args.task_id == 'all'
        cursor = conn.execute(
            """UPDATE Task
            SET status = 'PENDING',
                completed_at = NULL,
                claimed_at = NULL,
                worker_id = NULL
            WHERE status = 'FAILED'
            RETURNING id
            ; """,
            ())
        for row in cursor:
            print('Reset task_id', row[0])


def recover_crashed_tasks(timeout_seconds: int = 300):
    """
    The Reaper: Resets RUNNING tasks that have exceeded their time limit.
    Run this periodically in a separate thread or at the start of a worker.
    """
    with connect() as conn:
        cursor = conn.execute(f"""
            UPDATE Task
            SET status = 'PENDING', claimed_at = NULL
            WHERE status = 'RUNNING' 
              AND claimed_at < datetime('now', '-{timeout_seconds} seconds')
            RETURNING id;
        """)
        recovered = cursor.fetchall()
        if recovered:
            print(f"[Reaper] Recovered {len(recovered)} abandoned tasks.")


def task_completion_iter(
    worker_id=None,
    crashed_task_timeout_seconds=300,
    raise_on_failure=True,
    ) -> Iterator[dict]:
    if worker_id is None:
        worker_id = f'worker_{uuid.uuid4()!s}'


    recover_crashed_tasks(timeout_seconds=crashed_task_timeout_seconds)
    deadline_buffer = 10 # seconds
    while True:
        task_id, payload = claim_task(worker_id=worker_id)

        if task_id:
            print(f"Claimed Task {task_id}: {payload}")
            try:
                yield payload
                complete_task(task_id=task_id, worker_id=worker_id)
                print(f"Successfully completed Task {task_id}")

            except Exception as e:
                # Handled failures can be reset immediately or marked 'FAILED'
                print(f"Task {task_id} failed: {e}")
                with connect() as conn:
                    conn.execute("UPDATE Task SET status = 'FAILED' WHERE id = ?", (task_id,))
                if raise_on_failure:
                    raise e
        else:
            break


class NoRecord(Exception):
    pass


def by_id(table, conn=None, **kwargs):
    if conn is None:
        conn = connect()
    with conn:
        assert len(kwargs) == 1
        (key, val), = kwargs.items()
        try:
            query = f"SELECT {key} from {table} WHERE {key} = ? LIMIT 1;"
            cursor = conn.execute(query, (val,))
        except sqlite3.OperationalError as err:
            raise Exception(query) from  err
        for row in cursor:
            return row
        raise NoRecord()


def stable_hash(data: str, n_chars=8) -> str:
    """Returns a stable, cross-session SHA-256 hex string."""
    # 1. Encode the string to bytes
    encoded_data = data.encode('utf-8')

    # 2. Generate and return the hexadecimal digest
    return hashlib.shake_128(encoded_data).hexdigest(n_chars)


# create the top-level parser
parser = argparse.ArgumentParser(prog='planzero')
subparsers = parser.add_subparsers(help='subcommand help')

subparser = subparsers.add_parser('init')
subparser.set_defaults(func=main_init_db)

subparser = subparsers.add_parser('task_list')
#subparser.add_argument('--year', type=int, default=2005, help='year')
subparser.set_defaults(func=main_task_list)

subparser = subparsers.add_parser('task_complete_debug')
subparser.add_argument('--task-id', type=int, default=-1)
subparser.set_defaults(func=main_task_complete)

subparser = subparsers.add_parser('model_list')
#subparser.add_argument('--year', type=int, default=2005, help='year')
subparser.set_defaults(func=main_model_list)

subparser = subparsers.add_parser('component_type_list')
#subparser.add_argument('--year', type=int, default=2005, help='year')
subparser.set_defaults(func=main_component_type_list)

subparser = subparsers.add_parser('model_delete')
subparser.add_argument('ids', nargs='+')
subparser.set_defaults(func=main_model_delete)

subparser = subparsers.add_parser('task_reset')
subparser.add_argument('--task-id', type=str, default='all')
subparser.set_defaults(func=main_task_reset)

if __name__ == '__main__':
    import sys
    args = parser.parse_args()
    sys.exit(args.func(args))
