import jax
import jax.numpy as jnp


def stack_distributions(dists):
    """
    Stacks a list of NumPyro distributions of the same class into a 
    single vectorized distribution.
    """
    if not dists:
        raise ValueError("List of distributions cannot be empty.")
    
    # Ensure they are all the same type (e.g., all Normal, all Beta)
    dist_type = type(dists[0])
    orig_batch_shape = dists[0].batch_shape
    if not all(isinstance(d, dist_type) for d in dists):
        raise TypeError("All distributions must be of the same class.")

    if not all(d.batch_shape == orig_batch_shape for d in dists):
        raise NotImplementedError('heterogeneous batch shapes')

    # tree_map traverses the distributions, pulling out matching parameters 
    # (like 'loc' and 'scale'), stacking them, and reconstructing the object.
    stacked_dist = jax.tree_util.tree_map(lambda *leaves: jnp.stack(leaves), *dists)

    # 2. Correct the stale batch_shape metadata
    # The new batch shape is the length of the list, prepended to the original batch_shape
    new_batch_shape = (len(dists),) + orig_batch_shape
    
    assert hasattr(stacked_dist, "_batch_shape")
    assert stacked_dist._batch_shape == orig_batch_shape
    stacked_dist._batch_shape = new_batch_shape
    assert stacked_dist.batch_shape == new_batch_shape

    return stacked_dist
