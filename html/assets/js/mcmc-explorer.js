/*
 * Interactive dependency explorer for PlanZero inference groups.
 *
 * Each [data-mcmc-explorer] container carries a JSON payload produced by
 * planzero.dependency_explorer.  This script turns that payload into two
 * linked views:
 *
 *   Overview - an element-by-element dependency matrix.
 *   Trace    - a focused Cytoscape.js graph with an A -> B path finder.
 *
 * The script is deliberately dependency-free apart from Cytoscape.js, which
 * is loaded from a CDN immediately before this file.
 */
(function () {
    'use strict';

    if (window.__planzeroMcmcExplorerLoaded) {
        return;
    }
    window.__planzeroMcmcExplorerLoaded = true;

    const METHOD_ORDER = ['prepare', 'annual_step', 'postprocess'];
    const METHOD_LABELS = {
        prepare: 'Prep',
        annual_step: 'Annual Step',
        postprocess: 'Post',
    };
    const METHOD_COLORS = {
        prepare: '#2563eb',
        annual_step: '#0d9488',
        postprocess: '#7c3aed',
    };
    const KIND_LABELS = {
        general: 'General',
        annual: 'Annual',
        carry: 'Carry',
    };
    const KIND_COLORS = {
        general: '#64748b',
        annual: '#0284c7',
        carry: '#d97706',
    };
    const KIND_BADGE_CLASS = {
        general: 'mcmc-badge-general',
        annual: 'mcmc-badge-annual-kind',
        carry: 'mcmc-badge-carry',
    };
    const METHOD_BADGE_CLASS = {
        prepare: 'mcmc-badge-prep',
        annual_step: 'mcmc-badge-annual',
        postprocess: 'mcmc-badge-post',
    };
    const MAX_PATHS = 20;
    const MAX_PATH_SEARCH_STEPS = 50000;
    const TOOLTIP_VARIABLE_LIMIT = 12;
    const TRACE_ELEMENT_LIMIT = 3;

    function esc(value) {
        return String(value === undefined || value === null ? '' : value)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#39;');
    }

    function displayName(id) {
        return String(id === undefined || id === null ? '' : id).replace(/_/g, ' ');
    }

    function badge(text, className) {
        return '<span class="mcmc-badge ' + esc(className || '') + '">' +
            esc(text) + '</span>';
    }

    function initAll() {
        const containers = document.querySelectorAll('[data-mcmc-explorer]');
        for (let i = 0; i < containers.length; i++) {
            initExplorer(containers[i]);
        }
    }

    function initExplorer(container) {
        if (container.dataset.mcmcExplorerInitialized === '1') {
            return;
        }
        const dataScript = container.querySelector(
            'script.mcmc-explorer-data');
        if (!dataScript) {
            return;
        }

        let data;
        try {
            data = JSON.parse(dataScript.textContent);
        } catch (err) {
            container.innerHTML =
                '<div class="mcmc-explorer-error">' +
                'Could not parse the dependency explorer data: ' +
                esc(err.message) +
                '</div>';
            return;
        }

        container.dataset.mcmcExplorerInitialized = '1';

        const elements = Array.isArray(data.elements) ? data.elements : [];
        const variables = Array.isArray(data.variables) ? data.variables : [];
        const occurrences = Array.isArray(data.occurrences)
            ? data.occurrences : [];
        const properties = Array.isArray(data.properties)
            ? data.properties : [];
        const edges = Array.isArray(data.edges) ? data.edges : [];
        const elementsById = new Map();
        const variablesById = new Map();
        const occurrencesById = new Map();
        const propertiesById = new Map();

        for (const element of elements) {
            elementsById.set(element.id, element);
        }
        for (const variable of variables) {
            variablesById.set(variable.id, variable);
        }
        for (const occurrence of occurrences) {
            occurrencesById.set(occurrence.id, occurrence);
        }
        for (const property of properties) {
            propertiesById.set(property.id, property);
        }

        if (!elements.length) {
            container.innerHTML =
                '<div class="mcmc-explorer-error">' +
                'This inference group has no model elements to display.' +
                '</div>';
            return;
        }

        // Variable-level dependency graph.  Inside a single element, every
        // defined variable depends on every accessed variable.
        const varUp = new Map();
        const varDown = new Map();
        for (const element of elements) {
            const defines = element.define_variable_ids || [];
            const accesses = element.access_variable_ids || [];
            for (const defineId of defines) {
                if (!varUp.has(defineId)) {
                    varUp.set(defineId, new Set());
                }
                for (const accessId of accesses) {
                    if (accessId === defineId) {
                        continue;
                    }
                    varUp.get(defineId).add(accessId);
                    if (!varDown.has(accessId)) {
                        varDown.set(accessId, new Set());
                    }
                    varDown.get(accessId).add(defineId);
                }
            }
        }

        function elementLabel(id) {
            const element = elementsById.get(id);
            return element
                ? (element.display || element.pretty_name ||
                    displayName(element.id))
                : displayName(id);
        }

        function spreadsheetColumnCode(index) {
            let code = '';
            let remaining = index;
            do {
                code = String.fromCharCode(65 + (remaining % 26)) + code;
                remaining = Math.floor(remaining / 26) - 1;
            } while (remaining >= 0);
            return code;
        }

        function variableLabel(id) {
            const variable = variablesById.get(id);
            return variable ? (variable.display || displayName(variable.name))
                : displayName(id);
        }

        function readInitialState() {
            const params = new URLSearchParams(window.location.search);
            let focus = null;
            const focusParam = params.get('focus');
            if (focusParam) {
                const colon = focusParam.indexOf(':');
                const kind = colon === -1 ? '' : focusParam.slice(0, colon);
                const id = colon === -1 ? focusParam : focusParam.slice(
                    colon + 1);
                if (kind === 'el' && elementsById.has(id)) {
                    focus = { type: 'element', id: id };
                } else if (kind === 'var' && variablesById.has(id)) {
                    focus = { type: 'variable', id: id };
                }
            }

            let matrixScope = params.get('scope');
            if (matrixScope !== 'all' && matrixScope !== 'self') {
                matrixScope = 'cross';
            }

            const traceElements = [];
            for (const id of (params.get('trace') || '').split(',')) {
                if (elementsById.has(id) && traceElements.indexOf(id) === -1 &&
                        traceElements.length < TRACE_ELEMENT_LIMIT) {
                    traceElements.push(id);
                }
            }

            return {
                tab: params.get('tab') === 'trace' ? 'trace' : 'overview',
                traceElements: traceElements,
                focus: focus,
                matrixScope: matrixScope,
                matrixBackward: params.get('matrixBackward') !== '0',
            };
        }

        const initialState = readInitialState();
        const state = {
            tab: initialState.tab,
            traceElements: initialState.traceElements,
            focus: initialState.traceElements.length
                ? { type: 'element', id: initialState.traceElements[0] }
                : initialState.focus,
            matrixScope: initialState.matrixScope,
            matrixBackward: initialState.matrixBackward,
            matrixSort: 'group',
            matrixFilter: '',
            matrixCells: new Map(),
        };

        function matrixEdgeAllowed(edge) {
            if (!state.matrixBackward && edge.backward) {
                return false;
            }
            if (state.matrixScope === 'cross') {
                return !!edge.cross_element;
            }
            if (state.matrixScope === 'self') {
                return edge.source_element === edge.target_element;
            }
            return true;
        }

        /* -----------------------------------------------------------------
         * Shell
         * ---------------------------------------------------------------- */

        const summary = data.summary || {};
        const nElements = summary.n_elements || elements.length;
        const nProperties = summary.n_properties || properties.length;
        const nConnections = summary.n_property_connections === undefined
            ? new Set(edges.map(function (edge) {
                return edge.source_property + '\u0000' +
                    edge.target_property;
            })).size
            : summary.n_property_connections;
        const nCrossConnections = summary.n_cross_property_connections ===
            undefined
            ? new Set(edges.filter(function (edge) {
                return edge.cross_element;
            }).map(function (edge) {
                return edge.source_property + '\u0000' +
                    edge.target_property;
            })).size
            : summary.n_cross_property_connections;
        const statsHtml =
            '<div class="mcmc-explorer-stats">' +
                '<span class="mcmc-stat"><strong>' + nElements +
                    '</strong> elements</span>' +
                '<span class="mcmc-stat"><strong>' + nProperties +
                    '</strong> properties</span>' +
                '<span class="mcmc-stat"><strong>' + nConnections +
                    '</strong> connections</span>' +
                '<span class="mcmc-stat"><strong>' + nCrossConnections +
                    '</strong> cross-element</span>' +
            '</div>';

        container.innerHTML =
            '<div class="mcmc-explorer-tabs" role="tablist">' +
                '<button type="button" class="mcmc-explorer-tab" ' +
                    'role="tab" data-tab="overview" ' +
                    'aria-controls="mcmc-pane-overview">Overview</button>' +
                '<button type="button" class="mcmc-explorer-tab" ' +
                    'role="tab" data-tab="trace" ' +
                    'aria-controls="mcmc-pane-trace">Trace</button>' +
            '</div>' +
            '<div class="mcmc-explorer-pane" data-pane="overview" ' +
                'id="mcmc-pane-overview" role="tabpanel"></div>' +
            '<div class="mcmc-explorer-pane" data-pane="trace" ' +
                'id="mcmc-pane-trace" role="tabpanel" hidden></div>';

        const overviewPane = container.querySelector(
            '[data-pane="overview"]');
        const tracePane = container.querySelector('[data-pane="trace"]');
        const tabs = container.querySelector('.mcmc-explorer-tabs');

        const tabsSlot = document.querySelector('[data-mcmc-tabs-slot]');
        if (tabsSlot && !tabsSlot.querySelector('.mcmc-explorer-tabs')) {
            tabsSlot.appendChild(tabs);
        }

        const tooltip = document.createElement('div');
        tooltip.className = 'mcmc-explorer-tooltip';
        tooltip.hidden = true;
        container.appendChild(tooltip);

        /* -----------------------------------------------------------------
         * Tabs
         * ---------------------------------------------------------------- */

        function syncTabButtons() {
            const buttons = tabs.querySelectorAll('.mcmc-explorer-tab');
            for (const button of buttons) {
                const selected = button.dataset.tab === state.tab;
                button.setAttribute(
                    'aria-selected', selected ? 'true' : 'false');
                button.tabIndex = selected ? 0 : -1;
            }
            const panes = container.querySelectorAll('.mcmc-explorer-pane');
            for (const pane of panes) {
                pane.hidden = pane.dataset.pane !== state.tab;
            }
            const tabInput = document.querySelector('[data-mcmc-tab-input]');
            if (tabInput) {
                tabInput.value = state.tab;
            }
        }

        function syncHelpSections() {
            const helpRoot = container.closest('.model-explorer-page') ||
                document;
            const sections = helpRoot.querySelectorAll(
                '[data-mcmc-help] [data-help-tab]');
            for (const section of sections) {
                section.hidden = section.dataset.helpTab !== state.tab;
            }
        }

        function setTab(tab) {
            state.tab = tab;
            hideTooltip();
            syncTabButtons();
            syncHelpSections();
            if (tab === 'trace') {
                syncTraceControls();
                renderTrace();
            }
            updateUrl();
        }

        tabs.addEventListener('click', function (event) {
            const button = event.target.closest('[data-tab]');
            if (button) {
                setTab(button.dataset.tab);
            }
        });

        tabs.addEventListener('keydown', function (event) {
            if (event.key !== 'ArrowLeft' && event.key !== 'ArrowRight') {
                return;
            }
            const buttons = Array.prototype.slice.call(
                tabs.querySelectorAll('.mcmc-explorer-tab'));
            let index = buttons.indexOf(document.activeElement);
            if (index === -1) {
                return;
            }
            index += event.key === 'ArrowRight' ? 1 : -1;
            index = (index + buttons.length) % buttons.length;
            buttons[index].focus();
            setTab(buttons[index].dataset.tab);
            event.preventDefault();
        });

        /* -----------------------------------------------------------------
         * URL state
         * ---------------------------------------------------------------- */

        function updateUrl() {
            const params = new URLSearchParams(window.location.search);
            params.set('tab', state.tab);
            if (state.traceElements.length) {
                params.set('trace', state.traceElements.join(','));
            } else {
                params.delete('trace');
            }
            if (state.focus) {
                params.set(
                    'focus',
                    (state.focus.type === 'element' ? 'el:' : 'var:') +
                    state.focus.id);
            } else {
                params.delete('focus');
            }
            if (state.matrixScope !== 'cross') {
                params.set('scope', state.matrixScope);
            } else {
                params.delete('scope');
            }
            if (!state.matrixBackward) {
                params.set('matrixBackward', '0');
            } else {
                params.delete('matrixBackward');
            }
            const query = params.toString();
            const url = window.location.pathname +
                (query ? '?' + query : '') + window.location.hash;
            window.history.replaceState(null, '', url);
        }

        /* -----------------------------------------------------------------
         * Overview: matrix
         * ---------------------------------------------------------------- */

        overviewPane.innerHTML =
            '<div class="mcmc-toolbar">' +
                '<label class="mcmc-field">' +
                    '<span>Filter elements</span>' +
                    '<input type="search" data-role="matrixFilter" ' +
                        'placeholder="element or variable name\u2026">' +
                '</label>' +
                '<label class="mcmc-field">' +
                    '<span>Matrix scope</span>' +
                    '<select data-role="matrixScope">' +
                        '<option value="cross">Cross-element</option>' +
                        '<option value="all">All dependencies</option>' +
                        '<option value="self">Self dependencies</option>' +
                    '</select>' +
                '</label>' +
                '<label class="mcmc-field">' +
                    '<span>Sort</span>' +
                    '<select data-role="matrixSort">' +
                        '<option value="group">Model order</option>' +
                        '<option value="name">Name</option>' +
                        '<option value="dependency">Dependency count</option>' +
                    '</select>' +
                '</label>' +
                '<label class="mcmc-check">' +
                    '<input type="checkbox" data-role="matrixBackward" ' +
                        'checked> <span data-role="matrixBackwardLabel">' +
                        'hide carry accesses</span>' +
                '</label>' +
                '<div class="mcmc-toolbar-legend">' +
                    '<span class="mcmc-toolbar-legend-title">Legend</span>' +
                    '<span class="mcmc-legend-row">' +
                        '<span class="mcmc-legend-swatch" ' +
                            'style="background:rgba(24,191,239,0.34)">' +
                        '</span>' +
                        '<span>cell shade = distinct variables</span>' +
                    '</span>' +
                    '<span class="mcmc-legend-row">' +
                        '<span class="mcmc-legend-swatch" ' +
                            'style="border-color:#d97706">' +
                        '</span>' +
                        '<span>orange outline = self or carry link</span>' +
                    '</span>' +
                '</div>' +
                statsHtml +
            '</div>' +
            '<div class="mcmc-trace-hint" data-role="traceHint"></div>' +
            '<div class="mcmc-matrix-wrap" data-role="matrixWrap"></div>';

        const matrixWrap = overviewPane.querySelector(
            '[data-role="matrixWrap"]');
        const matrixFilterInput = overviewPane.querySelector(
            '[data-role="matrixFilter"]');

        function orderedElements() {
            const ordered = elements.slice();
            if (state.matrixSort === 'name') {
                ordered.sort(function (a, b) {
                    return elementLabel(a.id).localeCompare(
                        elementLabel(b.id));
                });
            } else if (state.matrixSort === 'dependency') {
                ordered.sort(function (a, b) {
                    const aScore =
                        (a.define_variable_ids || []).length +
                        (a.access_variable_ids || []).length;
                    const bScore =
                        (b.define_variable_ids || []).length +
                        (b.access_variable_ids || []).length;
                    if (aScore !== bScore) {
                        return bScore - aScore;
                    }
                    return elementLabel(a.id).localeCompare(
                        elementLabel(b.id));
                });
            }
            return ordered;
        }

        function propertyLabel(id) {
            const property = propertiesById.get(id);
            return property ? property.display : displayName(id);
        }

        function buildMatrixCells() {
            const cells = new Map();
            for (const edge of edges) {
                if (!matrixEdgeAllowed(edge)) {
                    continue;
                }
                const key = edge.source_element + '\u0000' +
                    edge.target_element;
                let cell = cells.get(key);
                if (!cell) {
                    cell = {
                        source: edge.source_element,
                        target: edge.target_element,
                        connections: new Map(),
                        variables: new Set(),
                        edgeIds: [],
                        backward: false,
                    };
                    cells.set(key, cell);
                }
                const connectionKey = edge.source_property + '\u0000' +
                    edge.target_property;
                if (!cell.connections.has(connectionKey)) {
                    cell.connections.set(connectionKey, {
                        source: edge.source_property,
                        target: edge.target_property,
                    });
                }
                cell.variables.add(edge.variable);
                cell.edgeIds.push(edge.id);
                if (edge.backward) {
                    cell.backward = true;
                }
            }
            return cells;
        }

        function renderMatrix() {
            hideTooltip();
            state.matrixCells = buildMatrixCells();

            const ordered = orderedElements();
            let maxCount = 1;
            for (const cell of state.matrixCells.values()) {
                maxCount = Math.max(maxCount, cell.connections.size);
            }

            if (!state.matrixCells.size) {
                matrixWrap.innerHTML =
                    '<div class="mcmc-matrix-empty-state">' +
                    'No dependencies match the current matrix scope.' +
                    '</div>';
                applyMatrixFilter();
                return;
            }

            const headerCells = ordered.map(function (element, index) {
                const code = spreadsheetColumnCode(index);
                const infoClass = element.site_model_element
                    ? ' mcmc-has-element-info' : '';
                const label = code + ' \u2014 ' + elementLabel(element.id);
                return '<th scope="col" class="mcmc-matrix-col' +
                    infoClass + '" data-element="' +
                    esc(element.id) + '" aria-label="' + esc(label) + '"' +
                    (infoClass
                        ? ''
                        : ' title="' + esc(label) + '"') + '>' +
                    '<span class="mcmc-matrix-col-code">' +
                    esc(code) +
                    '</span></th>';
            }).join('');

            const bodyRows = ordered.map(function (row, rowIndex) {
                const rowCode = spreadsheetColumnCode(rowIndex);
                const rowElement = elementsById.get(row.id);
                const infoClass = rowElement && rowElement.site_model_element
                    ? ' mcmc-has-element-info' : '';
                const cells = ordered.map(function (column, columnIndex) {
                    const key = row.id + '\u0000' + column.id;
                    const cell = state.matrixCells.get(key);
                    if (!cell) {
                        return '<td class="mcmc-matrix-empty" ' +
                            'aria-hidden="true">\u00b7</td>';
                    }
                    const count = cell.connections.size;
                    const level = Math.max(
                        1, Math.min(5, Math.ceil(count / maxCount * 5)));
                    const classes = [
                        'mcmc-matrix-cell',
                        'mcmc-level-' + level,
                    ];
                    if (row.id === column.id) {
                        classes.push('mcmc-is-self');
                    }
                    if (cell.backward) {
                        classes.push('mcmc-is-backward');
                    }
                    const label = rowCode + ' \u2192 ' +
                        spreadsheetColumnCode(columnIndex) + ': ' +
                        elementLabel(row.id) + ' \u2192 ' +
                        elementLabel(column.id) + ', ' + count +
                        ' property connection' + (count === 1 ? '' : 's');
                    return '<td class="' + classes.join(' ') + '" ' +
                        'tabindex="0" role="button" ' +
                        'data-source="' + esc(row.id) + '" ' +
                        'data-target="' + esc(column.id) + '" ' +
                        'aria-label="' + esc(label) + '">' +
                        '<span class="mcmc-matrix-count">' + count +
                        '</span></td>';
                }).join('');
                return '<tr data-element="' + esc(row.id) + '">' +
                    '<th scope="row" class="mcmc-matrix-index' +
                        infoClass + '" aria-label="' +
                        esc(rowCode + ' \u2014 ' + elementLabel(row.id)) +
                        '"' +
                        (infoClass
                            ? ''
                            : ' title="' +
                                esc(rowCode + ' \u2014 ' +
                                    elementLabel(row.id)) + '"') + '>' +
                        esc(rowCode) + '</th>' +
                    '<th scope="row" class="mcmc-matrix-row-heading' +
                        infoClass + '">' +
                    '<span class="mcmc-matrix-row-label">' +
                    esc(elementLabel(row.id)) +
                    '</span></th>' +
                    '<td class="mcmc-matrix-trace-cell">' +
                        '<input type="checkbox" data-trace-element="' +
                            esc(row.id) + '" aria-label="Trace ' +
                            esc(elementLabel(row.id)) + '">' +
                    '</td>' +
                    cells + '</tr>';
            }).join('');

            matrixWrap.innerHTML =
                '<table class="mcmc-matrix">' +
                    '<thead><tr>' +
                        '<th class="mcmc-matrix-corner-index" ' +
                            'scope="col"></th>' +
                        '<th class="mcmc-matrix-corner" scope="col">' +
                            '<span class="mcmc-matrix-corner-inner">' +
                                'producer \u2193<br>consumer \u2192' +
                            '</span>' +
                        '</th>' +
                        '<th class="mcmc-matrix-trace-head" scope="col" ' +
                            'title="Choose up to ' + TRACE_ELEMENT_LIMIT +
                            ' elements to show on the Trace tab">Trace</th>' +
                        headerCells +
                    '</tr></thead>' +
                    '<tbody>' + bodyRows + '</tbody>' +
                '</table>';

            applyMatrixFilter();
            syncTraceColumn();
        }

        /* -----------------------------------------------------------------
         * Overview: Trace column selection
         * ---------------------------------------------------------------- */

        function visibleAdjacency() {
            const adjacency = new Map();
            for (const edge of edges) {
                if (!matrixEdgeAllowed(edge) ||
                        edge.source_element === edge.target_element) {
                    continue;
                }
                for (const [from, to] of [
                    [edge.source_element, edge.target_element],
                    [edge.target_element, edge.source_element],
                ]) {
                    if (!adjacency.has(from)) {
                        adjacency.set(from, new Set());
                    }
                    adjacency.get(from).add(to);
                }
            }
            return adjacency;
        }

        function selectionIsConnected(ids, adjacency) {
            if (ids.length <= 1) {
                return true;
            }
            const members = new Set(ids);
            const seen = new Set([ids[0]]);
            const queue = [ids[0]];
            while (queue.length) {
                const current = queue.shift();
                for (const next of adjacency.get(current) || []) {
                    if (members.has(next) && !seen.has(next)) {
                        seen.add(next);
                        queue.push(next);
                    }
                }
            }
            return seen.size === members.size;
        }

        function syncTraceColumn() {
            const selected = state.traceElements;
            const adjacency = visibleAdjacency();
            const full = selected.length >= TRACE_ELEMENT_LIMIT;

            const boxes = matrixWrap.querySelectorAll(
                'input[data-trace-element]');
            for (const box of boxes) {
                const id = box.dataset.traceElement;
                const checked = selected.indexOf(id) !== -1;
                box.checked = checked;
                if (checked) {
                    const rest = selected.filter(function (other) {
                        return other !== id;
                    });
                    box.disabled = !selectionIsConnected(rest, adjacency);
                } else {
                    const touchesSelection = selected.some(function (other) {
                        return (adjacency.get(id) || new Set()).has(other);
                    });
                    box.disabled = full ||
                        (selected.length > 0 && !touchesSelection);
                }
            }

            const hint = overviewPane.querySelector(
                '[data-role="traceHint"]');
            if (hint) {
                hint.textContent = selected.length
                    ? 'Trace: ' + selected.length + ' of ' +
                        TRACE_ELEMENT_LIMIT + ' selected'
                    : 'Trace: tick up to ' + TRACE_ELEMENT_LIMIT +
                        ' elements to show them on the Trace tab';
            }
        }

        function setTraceElements(ids) {
            state.traceElements = ids.slice(0, TRACE_ELEMENT_LIMIT);
            state.focus = state.traceElements.length
                ? { type: 'element', id: state.traceElements[0] }
                : null;
            syncTraceColumn();
            if (state.tab === 'trace') {
                syncTraceControls();
                renderTrace();
            }
            updateUrl();
        }

        function toggleTraceElement(id, checked) {
            const adjacency = visibleAdjacency();
            const selected = state.traceElements.slice();
            const at = selected.indexOf(id);
            if (checked && at === -1) {
                const touchesSelection = selected.some(function (other) {
                    return (adjacency.get(id) || new Set()).has(other);
                });
                if (selected.length >= TRACE_ELEMENT_LIMIT ||
                        (selected.length && !touchesSelection)) {
                    syncTraceColumn();
                    return;
                }
                selected.push(id);
            } else if (!checked && at !== -1) {
                selected.splice(at, 1);
                if (!selectionIsConnected(selected, adjacency)) {
                    syncTraceColumn();
                    return;
                }
            }
            setTraceElements(selected);
        }

        matrixWrap.addEventListener('change', function (event) {
            const box = event.target.closest('input[data-trace-element]');
            if (box) {
                toggleTraceElement(box.dataset.traceElement, box.checked);
            }
        });

        function applyMatrixFilter() {
            const query = state.matrixFilter.trim().toLowerCase();
            const rows = matrixWrap.querySelectorAll('tbody tr');
            for (const row of rows) {
                const elementId = row.dataset.element || '';
                const rowMatch = !query ||
                    elementId.toLowerCase().indexOf(query) !== -1 ||
                    displayName(elementId).toLowerCase().indexOf(query) !== -1;
                row.classList.toggle('mcmc-dim', !rowMatch);

                const cells = row.querySelectorAll('td');
                for (const cell of cells) {
                    if (!cell.dataset.target) {
                        continue;
                    }
                    const targetId = cell.dataset.target;
                    const columnMatch = !query ||
                        targetId.toLowerCase().indexOf(query) !== -1 ||
                        displayName(targetId).toLowerCase().indexOf(query)
                            !== -1;
                    cell.classList.toggle(
                        'mcmc-dim', !rowMatch || !columnMatch);
                }
            }

            const headerCells = matrixWrap.querySelectorAll(
                'thead th[data-element]');
            for (const header of headerCells) {
                const elementId = header.dataset.element || '';
                const match = !query ||
                    elementId.toLowerCase().indexOf(query) !== -1 ||
                    displayName(elementId).toLowerCase().indexOf(query) !== -1;
                header.classList.toggle('mcmc-dim', !match);
            }
        }

        function clearMatrixRelations() {
            matrixWrap.querySelectorAll('.mcmc-is-hover-related')
                .forEach(function (element) {
                    element.classList.remove('mcmc-is-hover-related');
                });
        }

        function highlightMatrixRelations(cell) {
            clearMatrixRelations();
            const target = cell.dataset.target;
            if (!target) {
                return;
            }
            matrixWrap.querySelectorAll('thead th[data-element]')
                .forEach(function (header) {
                    if (header.dataset.element === target) {
                        header.classList.add('mcmc-is-hover-related');
                    }
                });
            matrixWrap.querySelectorAll('tbody tr[data-element]')
                .forEach(function (row) {
                    if (row.dataset.element !== target) {
                        return;
                    }
                    const name = row.querySelector(
                        '.mcmc-matrix-row-heading');
                    if (name) {
                        name.classList.add('mcmc-is-hover-related');
                    }
                });
        }

        function showCellTooltip(cell) {
            const key = cell.dataset.source + '\u0000' + cell.dataset.target;
            const cellData = state.matrixCells.get(key);
            if (!cellData) {
                return;
            }
            const connectionNames = Array.from(cellData.connections.values())
                .map(function (connection) {
                    return propertyLabel(connection.source) + ' \u2192 ' +
                        propertyLabel(connection.target);
                });
            const shown = connectionNames.slice(0, TOOLTIP_VARIABLE_LIMIT);
            const more = connectionNames.length - shown.length;
            const meta = connectionNames.length + ' property connection' +
                (connectionNames.length === 1 ? '' : 's') +
                (cellData.backward
                    ? ' \u00b7 includes carry / backward links' : '');

            tooltip.innerHTML =
                '<div class="mcmc-tooltip-title">' +
                    esc(elementLabel(cellData.source)) +
                    ' \u2192 ' +
                    esc(elementLabel(cellData.target)) +
                '</div>' +
                '<div class="mcmc-tooltip-meta">' + esc(meta) + '</div>' +
                '<ul class="mcmc-tooltip-list">' +
                    shown.map(function (name) {
                        return '<li>' + esc(name) + '</li>';
                    }).join('') +
                '</ul>' +
                (more > 0
                    ? '<div class="mcmc-tooltip-more">+ ' + more +
                        ' more</div>'
                    : '');
            tooltip.hidden = false;
        }

        function matrixHoverTarget(target) {
            const cell = target.closest('.mcmc-matrix-cell');
            if (cell) {
                return { anchor: cell, cell: cell };
            }
            const header = target.closest('thead th[data-element]');
            if (header) {
                return { anchor: header, elementId: header.dataset.element };
            }
            const rowCell = target.closest(
                '.mcmc-matrix-index, .mcmc-matrix-row-heading');
            if (rowCell) {
                const row = rowCell.closest('tbody tr[data-element]');
                if (row) {
                    return {
                        anchor: rowCell,
                        elementId: row.dataset.element,
                    };
                }
            }
            return null;
        }

        function showElementTooltip(elementId) {
            const element = elementsById.get(elementId);
            if (!element || !element.site_model_element) {
                hideTooltip();
                return;
            }
            const name = element.pretty_name || elementLabel(elementId);
            const description = element.short_description || '';
            tooltip.innerHTML =
                '<div class="mcmc-tooltip-title">' + esc(name) + '</div>' +
                (description
                    ? '<div class="mcmc-tooltip-description">' +
                        esc(description) + '</div>'
                    : '');
            tooltip.hidden = false;
        }

        function positionTooltip(clientX, clientY) {
            if (tooltip.hidden) {
                return;
            }
            const pad = 14;
            const rect = tooltip.getBoundingClientRect();
            let left = clientX + pad;
            let top = clientY + pad;
            if (left + rect.width > window.innerWidth - 8) {
                left = clientX - rect.width - pad;
            }
            if (top + rect.height > window.innerHeight - 8) {
                top = clientY - rect.height - pad;
            }
            left = Math.max(8, left);
            top = Math.max(8, top);
            tooltip.style.left = left + 'px';
            tooltip.style.top = top + 'px';
        }

        function hideTooltip() {
            tooltip.hidden = true;
        }

        function openMatrixCell(source, target) {
            const key = source + '\u0000' + target;
            const cell = state.matrixCells.get(key);
            if (!cell) {
                return;
            }
            setTraceElements(source === target ? [source] : [source, target]);
            setTab('trace');
        }

        matrixWrap.addEventListener('click', function (event) {
            const cell = event.target.closest('.mcmc-matrix-cell');
            if (cell) {
                openMatrixCell(cell.dataset.source, cell.dataset.target);
            }
        });

        matrixWrap.addEventListener('keydown', function (event) {
            if (event.key !== 'Enter' && event.key !== ' ') {
                return;
            }
            const cell = event.target.closest('.mcmc-matrix-cell');
            if (cell) {
                event.preventDefault();
                openMatrixCell(cell.dataset.source, cell.dataset.target);
            }
        });

        matrixWrap.addEventListener('mouseover', function (event) {
            const hover = matrixHoverTarget(event.target);
            if (!hover) {
                return;
            }
            if (hover.cell) {
                highlightMatrixRelations(hover.cell);
                showCellTooltip(hover.cell);
            } else {
                clearMatrixRelations();
                showElementTooltip(hover.elementId);
            }
            positionTooltip(event.clientX, event.clientY);
        });

        matrixWrap.addEventListener('mousemove', function (event) {
            if (!tooltip.hidden) {
                positionTooltip(event.clientX, event.clientY);
            }
        });

        matrixWrap.addEventListener('mouseout', function (event) {
            const hover = matrixHoverTarget(event.target);
            if (!hover) {
                return;
            }
            const related = event.relatedTarget;
            if (related && hover.anchor.contains(related)) {
                return;
            }
            clearMatrixRelations();
            hideTooltip();
        });

        matrixWrap.addEventListener('focusin', function (event) {
            const hover = matrixHoverTarget(event.target);
            if (!hover) {
                return;
            }
            if (hover.cell) {
                highlightMatrixRelations(hover.cell);
                showCellTooltip(hover.cell);
            } else {
                clearMatrixRelations();
                showElementTooltip(hover.elementId);
            }
            const rect = hover.anchor.getBoundingClientRect();
            positionTooltip(rect.left + rect.width / 2, rect.bottom);
        });

        matrixWrap.addEventListener('focusout', function () {
            clearMatrixRelations();
            hideTooltip();
        });

        /* -----------------------------------------------------------------
         * Overview: side panels
         * ---------------------------------------------------------------- */

        function syncMatrixBackwardLabel() {
            const label = overviewPane.querySelector(
                '[data-role="matrixBackwardLabel"]');
            if (label) {
                label.textContent = state.matrixBackward
                    ? 'hide carry accesses'
                    : 'show carry accesses';
            }
        }

        function syncOverviewControls() {
            const scopeSelect = overviewPane.querySelector(
                '[data-role="matrixScope"]');
            const sortSelect = overviewPane.querySelector(
                '[data-role="matrixSort"]');
            const backwardCheck = overviewPane.querySelector(
                '[data-role="matrixBackward"]');
            if (scopeSelect) {
                scopeSelect.value = state.matrixScope;
            }
            if (sortSelect) {
                sortSelect.value = state.matrixSort;
            }
            if (backwardCheck) {
                backwardCheck.checked = state.matrixBackward;
            }
            syncMatrixBackwardLabel();
            if (matrixFilterInput &&
                    matrixFilterInput.value !== state.matrixFilter) {
                matrixFilterInput.value = state.matrixFilter;
            }
        }

        overviewPane.addEventListener('change', function (event) {
            const role = event.target.dataset.role;
            if (role === 'matrixScope') {
                state.matrixScope = event.target.value;
                renderMatrix();
                updateUrl();
            } else if (role === 'matrixSort') {
                state.matrixSort = event.target.value;
                renderMatrix();
            } else if (role === 'matrixBackward') {
                state.matrixBackward = event.target.checked;
                syncMatrixBackwardLabel();
                renderMatrix();
                updateUrl();
            }
        });

        overviewPane.addEventListener('input', function (event) {
            if (event.target.dataset.role === 'matrixFilter') {
                state.matrixFilter = event.target.value;
                applyMatrixFilter();
            }
        });

        /* -----------------------------------------------------------------
         * Trace: controls
         * ---------------------------------------------------------------- */

        tracePane.innerHTML =
            '<div class="mcmc-trace-filter" data-role="filter" hidden>' +
            '</div>' +
            '<div class="mcmc-trace-layout">' +
                '<div class="mcmc-trace-graph-wrap">' +
                    '<div class="mcmc-trace-graph" data-role="graph">' +
                    '</div>' +
                '<div class="mcmc-trace-legend">' +
                        '<div class="mcmc-legend-row">' +
                            '<span class="mcmc-legend-swatch" ' +
                                'style="background:#2563eb"></span>' +
                            '<span>Prep</span>' +
                        '</div>' +
                        '<div class="mcmc-legend-row">' +
                            '<span class="mcmc-legend-swatch" ' +
                                'style="background:#0d9488"></span>' +
                            '<span>Annual Step</span>' +
                        '</div>' +
                        '<div class="mcmc-legend-row">' +
                            '<span class="mcmc-legend-swatch" ' +
                                'style="background:#7c3aed"></span>' +
                            '<span>Post</span>' +
                        '</div>' +
                        '<div class="mcmc-legend-row">' +
                            '<span class="mcmc-legend-swatch" ' +
                                'style="background:#ffffff;' +
                                'border-color:#0284c7"></span>' +
                            '<span>Annual variable</span>' +
                        '</div>' +
                        '<div class="mcmc-legend-row">' +
                            '<span class="mcmc-legend-swatch" ' +
                                'style="background:#ffffff;' +
                                'border-color:#d97706"></span>' +
                            '<span>Carry variable</span>' +
                        '</div>' +
                        '<div class="mcmc-legend-row">' +
                            '<span class="mcmc-legend-line dashed">' +
                            '</span>' +
                            '<span>Carry / backward link</span>' +
                        '</div>' +
                    '</div>' +
                '</div>' +
                '</div>' +
                '<aside class="mcmc-trace-sidebar">' +
                    '<div class="mcmc-details" data-role="details">' +
                    '</div>' +
                '</aside>' +
            '</div>';

        const filterBar = tracePane.querySelector('[data-role="filter"]');
        const graphEl = tracePane.querySelector('[data-role="graph"]');
        const detailsEl = tracePane.querySelector('[data-role="details"]');

        function option(value, label) {
            return '<option value="' + esc(value) + '">' +
                esc(label) + '</option>';
        }

        function populateTraceSelects() {
        }

        function syncTraceControls() {
            renderFilterBar();
        }

        function renderFilterBar() {
            if (!state.traceElements.length) {
                filterBar.hidden = true;
                filterBar.innerHTML = '';
                return;
            }
            const names = state.traceElements.map(elementLabel).join(
                ' \u00b7 ');
            filterBar.hidden = false;
            filterBar.innerHTML =
                '<span><strong>Selected:</strong> ' + esc(names) + '</span>' +
                '<button type="button" class="mcmc-button" ' +
                    'data-role="clearFilter">Clear</button>';
        }

        function setFocus(kind, id) {
            if (kind === 'element') {
                if (!elementsById.has(id)) {
                    return;
                }
                setTraceElements([id]);
                if (state.tab !== 'trace') {
                    setTab('trace');
                }
                return;
            }
            if (kind === 'variable' && variablesById.has(id)) {
                state.focus = { type: 'variable', id: id };
                renderTrace();
                updateUrl();
                if (state.tab !== 'trace') {
                    setTab('trace');
                }
            }
        }

        tracePane.addEventListener('click', function (event) {
            const role = event.target.dataset.role;
            if (role === 'clearFilter') {
                setTraceElements([]);
                return;
            }
            const link = event.target.closest('[data-focus-kind]');
            if (link) {
                setFocus(link.dataset.focusKind, link.dataset.focusId);
            }
        });

        /* -----------------------------------------------------------------
         * Trace: graph
         * ---------------------------------------------------------------- */

        const CY_STYLE = [
            {
                selector: 'node',
                style: {
                    'font-family': 'Helvetica, Arial, sans-serif',
                    'font-size': 10,
                    'label': 'data(label)',
                    'text-wrap': 'wrap',
                    'text-max-width': 150,
                    'text-valign': 'center',
                    'text-halign': 'center',
                    'background-color': 'data(color)',
                    'color': '#ffffff',
                    'border-width': 1,
                    'border-color': '#ffffff',
                    'shape': 'round-rectangle',
                    'padding': '7px',
                    'width': 'label',
                    'height': 'label',
                },
            },
            {
                selector: 'node[type="variable"]',
                style: {
                    'background-color': '#ffffff',
                    'border-color': 'data(color)',
                    'border-width': 2,
                    'color': '#1f2937',
                },
            },
            {
                selector: 'node.focus-node',
                style: {
                    'border-color': '#111827',
                    'border-width': 3,
                    'border-style': 'solid',
                },
            },
            {
                selector: 'node:selected',
                style: {
                    'border-color': '#111827',
                    'border-width': 3,
                },
            },
            {
                selector: 'edge',
                style: {
                    'width': 1.3,
                    'line-color': '#94a3b8',
                    'target-arrow-color': '#94a3b8',
                    'target-arrow-shape': 'triangle',
                    'arrow-scale': 0.8,
                    'curve-style': 'bezier',
                    'opacity': 0.9,
                },
            },
            {
                selector: 'edge[kind="define"]',
                style: {
                    'line-color': '#cbd5e1',
                    'target-arrow-color': '#cbd5e1',
                    'label': 'data(label)',
                    'font-size': 8,
                    'text-rotation': 'autorotate',
                    'text-background-color': '#fbfdfe',
                    'text-background-opacity': 0.88,
                    'text-background-padding': 2,
                },
            },
            {
                selector: 'edge[kind="access"]',
                style: {
                    'label': 'data(label)',
                    'font-size': 8,
                    'text-rotation': 'autorotate',
                    'text-background-color': '#fbfdfe',
                    'text-background-opacity': 0.88,
                    'text-background-padding': 2,
                },
            },
            {
                selector: 'edge[?backward]',
                style: {
                    'line-style': 'dashed',
                    'line-color': '#d97706',
                    'target-arrow-color': '#d97706',
                },
            },
            {
                selector: 'edge[kind="back"]',
                style: {
                    'line-style': 'dashed',
                    'line-color': '#d97706',
                    'target-arrow-color': '#d97706',
                    'label': 'data(label)',
                    'font-size': 8,
                    'text-rotation': 'autorotate',
                },
            },
            {
                selector: 'edge[kind="initial"], edge[kind="final"]',
                style: {
                    'line-color': '#94a3b8',
                    'target-arrow-color': '#94a3b8',
                    'label': 'data(label)',
                    'font-size': 8,
                    'text-rotation': 'autorotate',
                    'text-background-color': '#fbfdfe',
                    'text-background-opacity': 0.88,
                    'text-background-padding': 2,
                },
            },
            {
                selector: '.dimmed',
                style: {
                    'opacity': 0.12,
                },
            },
            {
                selector: '.highlighted',
                style: {
                    'opacity': 1,
                },
            },
        ];

        let cy = null;

        function methodNodeId(elementId, method) {
            return 'm:' + elementId + ':' + method;
        }

        function buildSelectionGraph() {
            const nodes = new Map();
            const edges = new Map();
            const selected = new Set(state.traceElements);
            if (!selected.size) {
                return { nodes: [], edges: [] };
            }

            const selectedOccurrences = occurrences.filter(function (o) {
                return selected.has(o.element);
            });

            function addMethodNode(elementId, method) {
                const id = methodNodeId(elementId, method);
                if (!nodes.has(id)) {
                    nodes.set(id, {
                        data: {
                            id: id,
                            type: 'method',
                            element: elementId,
                            method: method,
                            methodLabel: METHOD_LABELS[method] || method,
                            label: elementLabel(elementId) + '\n[' +
                                (METHOD_LABELS[method] || method) + ']',
                            color: METHOD_COLORS[method] || '#64748b',
                        },
                    });
                }
                return id;
            }

            function occurrenceKey(occurrence) {
                const ids = (occurrence.variable_ids || []).slice().sort();
                if (ids.length > 1) {
                    return 'd:' + ids.join(',');
                }
                return 'v:' + (ids[0] || occurrence.name);
            }

            function keyLabel(key, occurrence) {
                if (key.indexOf('d:') === 0) {
                    return (occurrence && occurrence.name
                        ? occurrence.name : key.slice(2)).replace(/_/g, ' ');
                }
                const vid = key.slice(2);
                const variable = variablesById.get(vid);
                return variable ? variableLabel(vid) :
                    ((occurrence && occurrence.name) || vid).replace(/_/g, ' ');
            }

            function addGeneralNode(key, occurrence) {
                const id = 'g:' + key;
                if (!nodes.has(id)) {
                    nodes.set(id, {
                        data: {
                            id: id,
                            type: 'variable',
                            varKind: 'general',
                            key: key,
                            variable: key.indexOf('v:') === 0 ? key.slice(2) : null,
                            label: keyLabel(key, occurrence),
                            color: KIND_COLORS.general,
                        },
                    });
                }
                return id;
            }

            function addAnnualNode(key, phase, occurrence) {
                const id = 'a:' + phase + ':' + key;
                if (!nodes.has(id)) {
                    nodes.set(id, {
                        data: {
                            id: id,
                            type: 'variable',
                            varKind: 'annual',
                            phase: phase,
                            key: key,
                            variable: key.indexOf('v:') === 0 ? key.slice(2) : null,
                            label: keyLabel(key, occurrence) + ' (' + phase + ')',
                            color: KIND_COLORS.annual,
                        },
                    });
                }
                return id;
            }

            function addCarryNode(key, phase, occurrence) {
                const id = 'c:' + phase + ':' + key;
                if (!nodes.has(id)) {
                    nodes.set(id, {
                        data: {
                            id: id,
                            type: 'variable',
                            varKind: 'carry',
                            phase: phase,
                            key: key,
                            variable: key.indexOf('v:') === 0 ? key.slice(2) : null,
                            label: keyLabel(key, occurrence) + ' (' + phase + ')',
                            color: KIND_COLORS.carry,
                        },
                    });
                }
                return id;
            }

            const edgePairs = new Set();

            function addEdge(id, source, target, kind, label, backward) {
                if (!source || !target) {
                    return;
                }
                const pair = source + '\0' + target;
                if (edgePairs.has(pair)) {
                    return;
                }
                edgePairs.add(pair);
                edges.set(id, {
                    data: {
                        id: id,
                        source: source,
                        target: target,
                        kind: kind,
                        label: label || '',
                        backward: !!backward,
                    },
                });
            }

            // Implied define_carry on annual_step when prepare defines carry
            // and the element overrides model_element_annual_step (spec).
            const impliedAnnualDefines = [];
            for (const occurrence of selectedOccurrences) {
                if (occurrence.method === 'prepare' &&
                        occurrence.direction === 'define' &&
                        occurrence.kind === 'carry') {
                    const element = elementsById.get(occurrence.element);
                    if (element && element.annual_step_overridden) {
                        impliedAnnualDefines.push({
                            element: occurrence.element,
                            method: 'annual_step',
                            direction: 'define',
                            kind: 'carry',
                            name: occurrence.name,
                            attr: 'model_element_define_carry',
                            variable_ids: occurrence.variable_ids || [],
                            _implied: true,
                            _sourceId: occurrence.id,
                        });
                    }
                }
            }

            const allActions = selectedOccurrences.concat(impliedAnnualDefines);

            // Ensure method nodes that are explicitly required exist.
            // (annual_step method node for implied define_carry is required
            // by spec; other actions create method nodes via rules below.)
            for (const action of impliedAnnualDefines) {
                addMethodNode(action.element, action.method);
            }

            function applyCarryLifecycle(methodId, key, occurrence, edgeBase, mode) {
                const initial = addCarryNode(key, 'initial', occurrence);
                const thisNode = addCarryNode(key, 'this', occurrence);
                const nextNode = addCarryNode(key, 'next', occurrence);
                const finalNode = addCarryNode(key, 'final', occurrence);
                addEdge(edgeBase + ':bi', initial, thisNode, 'initial', '', false);
                addEdge(edgeBase + ':bn', nextNode, thisNode, 'back', 'next→this', true);
                addEdge(edgeBase + ':nf', nextNode, finalNode, 'final', '', false);
                if (mode === 'access_this') {
                    addEdge(edgeBase + ':ma', thisNode, methodId, 'access',
                        occurrence.name, false);
                } else if (mode === 'access_next') {
                    addEdge(edgeBase + ':ma', nextNode, methodId, 'access',
                        occurrence.name, false);
                } else if (mode === 'define_next') {
                    addEdge(edgeBase + ':md', methodId, nextNode, 'define',
                        occurrence.name, false);
                }
            }

            for (const occurrence of allActions) {
                const key = occurrenceKey(occurrence);
                const attr = occurrence.attr || '';
                const method = occurrence.method;
                const direction = occurrence.direction;
                const kind = occurrence.kind;
                const name = occurrence.name;

                if (method === 'prepare' || method === 'postprocess') {
                    const methodId = addMethodNode(occurrence.element, method);
                    if (kind === 'carry' && direction === 'define') {
                        const initial = addCarryNode(key, 'initial', occurrence);
                        addEdge('d:' + (occurrence.id || occurrence._sourceId),
                            methodId, initial, 'define', name, false);
                        continue;
                    }
                    if (direction === 'access' && attr === 'model_element_access_val') {
                        const g = addGeneralNode(key, occurrence);
                        addEdge('a:' + occurrence.id, g, methodId, 'access', name, false);
                        continue;
                    }
                    if (direction === 'access' && attr === 'model_element_access_annual') {
                        const all = addAnnualNode(key, 'all', occurrence);
                        addEdge('a:' + occurrence.id, all, methodId, 'access', name, false);
                        continue;
                    }
                    if (direction === 'define' && attr === 'model_element_define') {
                        const g = addGeneralNode(key, occurrence);
                        addEdge('d:' + occurrence.id, methodId, g, 'define', name, false);
                        continue;
                    }
                    if (direction === 'define' && attr === 'model_element_define_annual') {
                        const all = addAnnualNode(key, 'all', occurrence);
                        addEdge('d:' + occurrence.id, methodId, all, 'define', name, false);
                        continue;
                    }
                    // Fallback for any other val/annual general cases.
                    if (kind === 'general') {
                        const g = addGeneralNode(key, occurrence);
                        if (direction === 'define') {
                            addEdge('d:' + occurrence.id, methodId, g, 'define', name, false);
                        } else {
                            addEdge('a:' + occurrence.id, g, methodId, 'access', name, false);
                        }
                    } else if (kind === 'annual') {
                        const all = addAnnualNode(key, 'all', occurrence);
                        if (direction === 'define') {
                            addEdge('d:' + occurrence.id, methodId, all, 'define', name, false);
                        } else {
                            addEdge('a:' + occurrence.id, all, methodId, 'access', name, false);
                        }
                    }
                    continue;
                }

                if (method === 'annual_step') {
                    const methodId = addMethodNode(occurrence.element, method);
                    if (attr === 'model_element_access_this_carry') {
                        applyCarryLifecycle(methodId, key, occurrence,
                            'lc:' + occurrence.id, 'access_this');
                        continue;
                    }
                    if (attr === 'model_element_access_next_carry') {
                        applyCarryLifecycle(methodId, key, occurrence,
                            'lc:' + occurrence.id, 'access_next');
                        continue;
                    }
                    if (attr === 'model_element_define_carry') {
                        const base = occurrence._implied
                            ? ('lc:implied:' + occurrence.element + ':' + key)
                            : ('lc:' + occurrence.id);
                        applyCarryLifecycle(methodId, key, occurrence,
                            base, 'define_next');
                        continue;
                    }
                    if (attr === 'model_element_access_annual') {
                        const all = addAnnualNode(key, 'all', occurrence);
                        const thisNode = addAnnualNode(key, 'this', occurrence);
                        addEdge('aa:ba:' + occurrence.id, all, thisNode, 'annual', '', false);
                        addEdge('aa:ma:' + occurrence.id, thisNode, methodId, 'access', name, false);
                        continue;
                    }
                    if (attr === 'model_element_define_annual') {
                        const all = addAnnualNode(key, 'all', occurrence);
                        const thisNode = addAnnualNode(key, 'this', occurrence);
                        addEdge('ad:am:' + occurrence.id, methodId, thisNode, 'define', name, false);
                        addEdge('ad:tb:' + occurrence.id, thisNode, all, 'annual', '', false);
                        continue;
                    }
                    if (attr === 'model_element_access_val') {
                        const g = addGeneralNode(key, occurrence);
                        addEdge('a:' + occurrence.id, g, methodId, 'access', name, false);
                        continue;
                    }
                    // Fallbacks (should be rare/forbidden cases handled gracefully)
                    if (kind === 'annual') {
                        const all = addAnnualNode(key, 'all', occurrence);
                        if (direction === 'define') {
                            addEdge('d:' + occurrence.id, methodId, all, 'define', name, false);
                        } else {
                            addEdge('a:' + occurrence.id, all, methodId, 'access', name, false);
                        }
                    } else if (kind === 'general') {
                        const g = addGeneralNode(key, occurrence);
                        if (direction === 'define') {
                            addEdge('d:' + occurrence.id, methodId, g, 'define', name, false);
                        } else {
                            addEdge('a:' + occurrence.id, g, methodId, 'access', name, false);
                        }
                    }
                }
            }

                return {
                nodes: Array.from(nodes.values()),
                edges: Array.from(edges.values()),
            };
        }

        /*
         * Grid layout for the Trace graph.  The rules are in
         * design/detailed_model_element_layout.txt.  Rows are assigned in
         * section order A, B, C, D, E; columns go directly below a node's
         * first placed neighbour when that grid point is free, otherwise to
         * the next free grid point on the right.
         */
        const GRID_SPACING_X = 200;
        const GRID_SPACING_Y = 110;
        const SECTION_RANK = { A: 0, B: 1, C: 2, D: 3, E: 4 };
        const METHOD_SECTION = {
            prepare: 'A',
            annual_step: 'C',
            postprocess: 'E',
        };

        function computeGridLayout(graphModel) {
            const nodeData = new Map();
            for (const node of graphModel.nodes) {
                nodeData.set(node.data.id, node.data);
            }
            const inEdges = new Map();
            const outEdges = new Map();
            for (const id of nodeData.keys()) {
                inEdges.set(id, []);
                outEdges.set(id, []);
            }
            for (const edge of graphModel.edges) {
                if (edge.data.kind === 'back') {
                    continue;
                }
                inEdges.get(edge.data.target).push(edge.data);
                outEdges.get(edge.data.source).push(edge.data);
            }

            // Kahn's algorithm on the graph without next->this back edges.
            const indegree = new Map();
            const queue = [];
            for (const id of nodeData.keys()) {
                indegree.set(id, inEdges.get(id).length);
                if (indegree.get(id) === 0) {
                    queue.push(id);
                }
            }
            const order = [];
            for (let i = 0; i < queue.length; i++) {
                const id = queue[i];
                order.push(id);
                for (const edge of outEdges.get(id)) {
                    const deg = indegree.get(edge.target) - 1;
                    indegree.set(edge.target, deg);
                    if (deg === 0) {
                        queue.push(edge.target);
                    }
                }
            }
            if (order.length !== nodeData.size) {
                throw new Error('the trace graph contains a cycle');
            }

            const isMethod = (id) => nodeData.get(id).type === 'method';
            const isAnnualThis = (id) => {
                const d = nodeData.get(id);
                return d.type === 'variable' && d.varKind === 'annual' &&
                    d.phase === 'this';
            };
            const definerOf = (id) => {
                const edge = inEdges.get(id).find(
                    (e) => e.kind === 'define' && isMethod(e.source));
                return edge ? edge.source : null;
            };

            function earliestConsumerSection(id) {
                let best = null;
                for (const edge of outEdges.get(id)) {
                    const section = sectionOf(edge.target);
                    if (best === null ||
                            SECTION_RANK[section] < SECTION_RANK[best]) {
                        best = section;
                    }
                }
                return best || 'A';
            }

            // Section of each node.  FEED marks inputs that feed a B node;
            // they are placed on the row immediately above section B.
            function sectionOf(id) {
                const d = nodeData.get(id);
                if (d.type === 'method') {
                    return METHOD_SECTION[d.method];
                }
                const definer = definerOf(id);
                if (d.varKind === 'carry') {
                    if (d.phase === 'this') {
                        return 'B';
                    }
                    if (d.phase === 'final') {
                        return 'D';
                    }
                    if (d.phase === 'next') {
                        return 'C';
                    }
                    return definer ? 'A' : 'FEED';
                }
                if (isAnnualThis(id)) {
                    return definer ? 'C' : 'B';
                }
                if (definer) {
                    return METHOD_SECTION[nodeData.get(definer).method];
                }
                if (inEdges.get(id).some((e) => isAnnualThis(e.source))) {
                    return 'D';
                }
                if (outEdges.get(id).some((e) => isAnnualThis(e.target))) {
                    return 'FEED';
                }
                return earliestConsumerSection(id);
            }

            const sections = new Map();
            for (const id of order) {
                sections.set(id, sectionOf(id));
            }
            const members = (section) => order.filter(
                (id) => sections.get(id) === section);

            // Rows.  A method sits one row below its base (the section's
            // non-method first row); every other node sits at or below the
            // row of its inputs plus one.
            const level = new Map();
            const methodRows = new Set();
            function place(id, base) {
                let row = isMethod(id) ? base + 1 : base;
                for (const edge of inEdges.get(id)) {
                    if (level.has(edge.source)) {
                        row = Math.max(row, level.get(edge.source) + 1);
                    }
                }
                level.set(id, row);
                if (isMethod(id)) {
                    methodRows.add(row);
                }
            }
            const maxRow = (ids) => Math.max(...ids.map((id) => level.get(id)));
            function sectionEnd(lastRow) {
                const endRow = methodRows.has(lastRow) ? lastRow + 1 : lastRow;
                return endRow + 1;
            }

            let cursor = 0;

            const sectionA = members('A');
            for (const id of sectionA) {
                place(id, 0);
            }
            if (sectionA.length) {
                cursor = sectionEnd(maxRow(sectionA));
            }

            const sectionB = members('B');
            const feeders = members('FEED');
            if (sectionB.length) {
                let row = cursor;
                if (feeders.length && row === 0) {
                    row = 1;
                }
                for (const id of sectionB) {
                    level.set(id, row);
                }
                for (const id of feeders) {
                    level.set(id, row - 1);
                }
                cursor = row + 1;
            }

            const sectionC = members('C');
            const cBase = cursor;
            for (const id of sectionC) {
                place(id, cBase);
            }
            if (sectionC.length) {
                cursor = sectionEnd(maxRow(sectionC));
            }

            const sectionD = members('D');
            if (sectionD.length) {
                for (const id of sectionD) {
                    level.set(id, cursor);
                }
                cursor += 1;
            }

            const sectionE = members('E');
            for (const id of sectionE) {
                place(id, cursor);
            }

            for (const id of nodeData.keys()) {
                if (!level.has(id)) {
                    throw new Error('unplaced trace node ' + id);
                }
            }

            // Columns.  Directly below the first placed neighbour when that
            // grid point is free, otherwise the next free point on the right.
            const cols = new Map();
            const taken = new Map();
            function claimColumn(row, preferred) {
                if (!taken.has(row)) {
                    taken.set(row, new Set());
                }
                const used = taken.get(row);
                let col = preferred === undefined ? 0 : preferred;
                while (used.has(col)) {
                    col++;
                }
                used.add(col);
                return col;
            }
            function placeColumn(id, neighbourIds) {
                let preferred;
                for (const neighbour of neighbourIds) {
                    if (cols.has(neighbour)) {
                        preferred = cols.get(neighbour);
                        break;
                    }
                }
                cols.set(id, claimColumn(level.get(id), preferred));
            }
            // Top to bottom, so each node sees its neighbour above (or, for
            // feeders, its neighbour below) already placed.  Within a row the
            // carry-this/carry-final pairs go first, feeders last.
            const priorityOf = (id) => {
                if (sections.get(id) === 'FEED') {
                    return 2;
                }
                const d = nodeData.get(id);
                const paired = d.varKind === 'carry' &&
                    (d.phase === 'this' || d.phase === 'final');
                return paired ? 0 : 1;
            };
            const columnOrder = order
                .map((id, index) => ({ id, index }))
                .sort((a, b) => level.get(a.id) - level.get(b.id) ||
                    priorityOf(a.id) - priorityOf(b.id) ||
                    a.index - b.index)
                .map((item) => item.id);
            for (const id of columnOrder) {
                const neighbours = sections.get(id) === 'FEED'
                    ? outEdges.get(id).map((e) => e.target)
                    : inEdges.get(id).map((e) => e.source);
                placeColumn(id, neighbours);
            }

            const positions = {};
            for (const id of nodeData.keys()) {
                positions[id] = {
                    x: cols.get(id) * GRID_SPACING_X,
                    y: level.get(id) * GRID_SPACING_Y,
                };
            }
            return positions;
        }

        function renderGraph(graphModel) {
            if (cy) {
                cy.destroy();
                cy = null;
            }
            graphEl.innerHTML = '';

            if (!window.cytoscape) {
                graphEl.innerHTML =
                    '<div class="mcmc-trace-graph-message">' +
                    'Cytoscape.js could not be loaded. Check the network ' +
                    'connection or the CDN script.' +
                    '</div>';
                return;
            }

            if (!graphModel.nodes.length) {
                graphEl.innerHTML =
                    '<div class="mcmc-trace-graph-message">' +
                    (state.traceElements.length
                        ? 'The selected elements have no define-decorated ' +
                            'methods to show.'
                        : 'Tick up to ' + TRACE_ELEMENT_LIMIT +
                            ' model elements in the Trace column of the ' +
                            'Overview table to show them here.') +
                    '</div>';
                return;
            }

            // Compute grid-based layout positions
            const positions = computeGridLayout(graphModel);

            // Add positions to node data
            const elementsForCy = graphModel.nodes.concat(graphModel.edges);
            for (const el of elementsForCy) {
                if (!el.data.source && positions[el.data.id]) {
                    el.position = positions[el.data.id];
                }
            }

            cy = window.cytoscape({
                container: graphEl,
                elements: elementsForCy,
                style: CY_STYLE,
                layout: {
                    name: 'preset',
                    positions: function(node) {
                        return positions[node.data('id')] || { x: 0, y: 0 };
                    },
                    fit: true,
                    padding: 24,
                    animate: false,
                },
                minZoom: 0.15,
                maxZoom: 2.5,
                wheelSensitivity: 0.2,
                boxSelectionEnabled: false,
            });

            cy.nodes().forEach(function (node) {
                const nodeData = node.data();
                if (state.focus && state.focus.type === 'element' &&
                        nodeData.type === 'method' &&
                        nodeData.element === state.focus.id) {
                    node.addClass('focus-node');
                }
                if (state.focus && state.focus.type === 'variable' &&
                        nodeData.type === 'variable' &&
                        nodeData.variable === state.focus.id) {
                    node.addClass('focus-node');
                }
            });

            cy.on('tap', 'node', function (event) {
                const nodeData = event.target.data();
                if (nodeData.type === 'method') {
                    showMethodDetails(nodeData.element, nodeData.method);
                } else if (nodeData.type === 'variable') {
                    if (nodeData.variable && variablesById.has(nodeData.variable)) {
                        showVariableDetails(nodeData.variable);
                    } else {
                        detailsEl.innerHTML =
                            '<h3>' + esc(nodeData.label || 'Variable') + '</h3>' +
                            '<div class="mcmc-details-meta">' +
                            esc(nodeData.varKind || 'variable') +
                            (nodeData.phase ? ' · ' + esc(nodeData.phase) : '') +
                            '</div>';
                    }
                }
            });

            cy.on('tap', function (event) {
                if (event.target === cy) {
                    cy.elements().unselect();
                    showFocusDetails();
                }
            });

            cy.on('mouseover', 'node', function (event) {
                const node = event.target;
                cy.elements().addClass('dimmed');
                const neighborhood = node.closedNeighborhood();
                neighborhood.removeClass('dimmed').addClass('highlighted');
                node.addClass('highlighted');
            });

            cy.on('mouseout', 'node', function () {
                cy.elements().removeClass('dimmed highlighted');
            });
        }

        /* -----------------------------------------------------------------
         * Trace: details
         * ---------------------------------------------------------------- */

        function occurrenceItem(occurrenceId, direction) {
            const occurrence = occurrencesById.get(occurrenceId);
            if (!occurrence) {
                return '';
            }
            const variableLinks = (occurrence.variable_ids || []).map(
                function (variableId) {
                    return '<span class="mcmc-link" ' +
                        'data-focus-kind="variable" ' +
                        'data-focus-id="' + esc(variableId) + '">' +
                        esc(variableLabel(variableId)) + '</span>';
                }).join(', ');
            const methodLabel = METHOD_LABELS[occurrence.method] ||
                occurrence.method;
            const backward = occurrence.backward
                ? ' ' + badge('carry / backward', 'mcmc-badge-backward')
                : '';
            return '<li class="mcmc-details-item">' +
                '<strong>' + esc(occurrence.name) + '</strong>' +
                ' ' + badge(methodLabel,
                    METHOD_BADGE_CLASS[occurrence.method] || '') +
                backward +
                '<br>' +
                (variableLinks || '<span class="mcmc-details-meta">' +
                    'no resolved variables</span>') +
                '</li>';
        }

        function listHtml(items, emptyText) {
            if (!items.length) {
                return '<p class="mcmc-details-meta">' + esc(emptyText) +
                    '</p>';
            }
            return '<ul class="mcmc-details-list">' +
                items.join('') + '</ul>';
        }

        function elementLink(elementId) {
            return '<span class="mcmc-link" data-focus-kind="element" ' +
                'data-focus-id="' + esc(elementId) + '">' +
                esc(elementLabel(elementId)) + '</span>';
        }

        function variableLink(variableId) {
            return '<span class="mcmc-link" data-focus-kind="variable" ' +
                'data-focus-id="' + esc(variableId) + '">' +
                esc(variableLabel(variableId)) + '</span>';
        }

        function showDetailsEmpty() {
            detailsEl.innerHTML =
                '<p class="mcmc-details-empty">Click a node in the graph ' +
                'to inspect its declarations. Use the matrix cell click ' +
                'to jump from an overview dependency to its trace.</p>';
        }

        function showElementDetails(elementId) {
            const element = elementsById.get(elementId);
            if (!element) {
                showDetailsEmpty();
                return;
            }
            const sections = METHOD_ORDER.map(function (method) {
                const methodData = element.methods &&
                    element.methods[method];
                if (!methodData) {
                    return '';
                }
                const defines = (methodData.defines || []).map(
                    function (occurrenceId) {
                        return occurrenceItem(occurrenceId, 'define');
                    });
                const accesses = (methodData.accesses || []).map(
                    function (occurrenceId) {
                        return occurrenceItem(occurrenceId, 'access');
                    });
                if (!defines.length && !accesses.length) {
                    return '';
                }
                return '<h4>' + badge(
                        METHOD_LABELS[method] || method,
                        METHOD_BADGE_CLASS[method] || '') +
                    '</h4>' +
                    '<div class="mcmc-details-meta">Defines</div>' +
                    listHtml(defines, 'No definitions') +
                    '<div class="mcmc-details-meta">Accesses</div>' +
                    listHtml(accesses, 'No accesses');
            }).join('');

            detailsEl.innerHTML =
                '<h3>' + esc(elementLabel(element.id)) + '</h3>' +
                '<div class="mcmc-details-meta">' +
                    '<code>' + esc(element.id) + '</code><br>' +
                    (element.define_variable_ids || []).length +
                    ' defined variables \u00b7 ' +
                    (element.access_variable_ids || []).length +
                    ' accessed variables' +
                '</div>' +
                sections;
        }

        function showMethodDetails(elementId, method) {
            const element = elementsById.get(elementId);
            const methodData = element && element.methods &&
                element.methods[method];
            if (!element || !methodData) {
                showDetailsEmpty();
                return;
            }
            const defines = (methodData.defines || []).map(
                function (occurrenceId) {
                    return occurrenceItem(occurrenceId, 'define');
                });
            const accesses = (methodData.accesses || []).map(
                function (occurrenceId) {
                    return occurrenceItem(occurrenceId, 'access');
                });
            detailsEl.innerHTML =
                '<h3>' + esc(elementLabel(element.id)) + '</h3>' +
                '<div class="mcmc-details-meta">' +
                    badge(METHOD_LABELS[method] || method,
                        METHOD_BADGE_CLASS[method] || '') +
                    ' <code>' + esc(element.id) + '</code>' +
                '</div>' +
                '<h4>Defines</h4>' +
                listHtml(defines, 'No definitions') +
                '<h4>Accesses</h4>' +
                listHtml(accesses, 'No accesses');
        }

        function showVariableDetails(variableId) {
            const variable = variablesById.get(variableId);
            if (!variable) {
                showDetailsEmpty();
                return;
            }
            const defines = (variable.defines || []).map(
                function (occurrenceId) {
                    return occurrenceItem(occurrenceId, 'define');
                });
            const accesses = (variable.accesses || []).map(
                function (occurrenceId) {
                    return occurrenceItem(occurrenceId, 'access');
                });
            const producers = (variable.define_elements || []).map(
                elementLink);
            const consumers = (variable.access_elements || []).map(
                elementLink);
            const upstream = Array.from(varUp.get(variableId) || [])
                .slice(0, 12).map(variableLink);
            const downstream = Array.from(varDown.get(variableId) || [])
                .slice(0, 12).map(variableLink);
            const aliases = variable.aliases || [];

            detailsEl.innerHTML =
                '<h3>' + esc(variableLabel(variable.id)) + '</h3>' +
                '<div class="mcmc-details-meta">' +
                    badge(KIND_LABELS[variable.kind] || variable.kind,
                        KIND_BADGE_CLASS[variable.kind] || '') +
                    (variable.cross_element
                        ? ' ' + badge('cross-element', '') : '') +
                    (variable.self_element
                        ? ' ' + badge('self', '') : '') +
                '</div>' +
                (aliases.length > 1
                    ? '<div class="mcmc-details-meta">Aliases: ' +
                        esc(aliases.join(', ')) + '</div>'
                    : '') +
                '<h4>Defined by</h4>' +
                listHtml(producers, 'No producers') +
                '<h4>Accessed by</h4>' +
                listHtml(consumers, 'No consumers') +
                '<h4>Depends on (variables)</h4>' +
                listHtml(upstream, 'No upstream variables in this group') +
                '<h4>Feeds into (variables)</h4>' +
                listHtml(downstream, 'No downstream variables in this group') +
                '<h4>Define declarations</h4>' +
                listHtml(defines, 'No define declarations') +
                '<h4>Access declarations</h4>' +
                listHtml(accesses, 'No access declarations');
        }

        function showFocusDetails() {
            if (!state.focus) {
                showDetailsEmpty();
                return;
            }
            if (state.focus.type === 'variable') {
                showVariableDetails(state.focus.id);
            } else {
                showElementDetails(state.focus.id);
            }
        }

        /* -----------------------------------------------------------------
         * Trace: path finder
         * ---------------------------------------------------------------- */






        /* -----------------------------------------------------------------
         * Trace: rendering
         * ---------------------------------------------------------------- */

        function renderTrace() {
            const graphModel = buildSelectionGraph();
            renderGraph(graphModel);
            showFocusDetails();
            renderFilterBar();
        }

        /* -----------------------------------------------------------------
         * Initialization
         * ---------------------------------------------------------------- */

        populateTraceSelects();
        syncOverviewControls();
        syncTabButtons();
        syncHelpSections();
        renderMatrix();

        if (state.tab === 'trace') {
            syncTraceControls();
            renderTrace();
        } else {
            showDetailsEmpty();
        }
        updateUrl();
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initAll);
    } else {
        initAll();
    }
})();
