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

        const edgesBySource = new Map();
        const edgesByTarget = new Map();
        for (const edge of edges) {
            if (!edgesBySource.has(edge.source_element)) {
                edgesBySource.set(edge.source_element, []);
            }
            edgesBySource.get(edge.source_element).push(edge);

            if (!edgesByTarget.has(edge.target_element)) {
                edgesByTarget.set(edge.target_element, []);
            }
            edgesByTarget.get(edge.target_element).push(edge);
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

        function variableOptionLabel(variable) {
            const producers = (variable.define_elements || [])
                .map(elementLabel)
                .join(', ');
            let label = variableLabel(variable.id) +
                ' \u2014 ' + (KIND_LABELS[variable.kind] || 'Variable');
            if (producers) {
                label += ' \u2014 defined by ' + producers;
            }
            return label;
        }

        function defaultFocus() {
            let bestId = elements[0].id;
            let bestScore = -1;
            for (const element of elements) {
                const score =
                    (element.define_variable_ids || []).length +
                    (element.access_variable_ids || []).length;
                if (score > bestScore) {
                    bestScore = score;
                    bestId = element.id;
                }
            }
            return { type: 'element', id: bestId };
        }

        function readInitialState() {
            const params = new URLSearchParams(window.location.search);
            let focus = defaultFocus();
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

            let depth = parseInt(params.get('depth'), 10);
            if (!(depth >= 1 && depth <= 3)) {
                depth = 1;
            }

            let direction = params.get('direction');
            if (direction !== 'upstream' && direction !== 'downstream') {
                direction = 'both';
            }

            let matrixScope = params.get('scope');
            if (matrixScope !== 'all' && matrixScope !== 'self') {
                matrixScope = 'cross';
            }

            return {
                tab: params.get('tab') === 'trace' ? 'trace' : 'overview',
                focus: focus,
                depth: depth,
                direction: direction,
                showVariables: params.get('vars') !== '0',
                allowBackward: params.get('backward') !== '0',
                allowSelf: params.get('self') === '1',
                matrixScope: matrixScope,
                matrixBackward: params.get('matrixBackward') !== '0',
            };
        }

        const initialState = readInitialState();
        const state = {
            tab: initialState.tab,
            focus: initialState.focus,
            depth: initialState.depth,
            direction: initialState.direction,
            showVariables: initialState.showVariables,
            allowBackward: initialState.allowBackward,
            allowSelf: initialState.allowSelf,
            matrixScope: initialState.matrixScope,
            matrixBackward: initialState.matrixBackward,
            matrixSort: 'group',
            matrixFilter: '',
            matrixCells: new Map(),
            traceFilter: null,
            pathResults: null,
            pathFrom: null,
            pathTo: null,
        };

        function edgeAllowed(edge, allowBackward, allowSelf) {
            const backwardOk = allowBackward === undefined
                ? state.allowBackward : allowBackward;
            const selfOk = allowSelf === undefined
                ? state.allowSelf : allowSelf;
            if (!selfOk && edge.source_element === edge.target_element) {
                return false;
            }
            if (!backwardOk && edge.backward) {
                return false;
            }
            return true;
        }

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

        function elementScope(startId, depth, direction) {
            const scope = new Set([startId]);
            let frontier = new Set([startId]);

            for (let hop = 0; hop < depth; hop++) {
                const next = new Set();
                for (const id of frontier) {
                    if (direction !== 'downstream') {
                        for (const edge of edgesByTarget.get(id) || []) {
                            if (edgeAllowed(edge)) {
                                next.add(edge.source_element);
                            }
                        }
                    }
                    if (direction !== 'upstream') {
                        for (const edge of edgesBySource.get(id) || []) {
                            if (edgeAllowed(edge)) {
                                next.add(edge.target_element);
                            }
                        }
                    }
                }
                let added = false;
                for (const id of next) {
                    if (!scope.has(id)) {
                        scope.add(id);
                        added = true;
                    }
                }
                if (!added) {
                    break;
                }
                frontier = next;
            }
            return scope;
        }

        function variableScope(startId, depth, direction) {
            const scope = new Set([startId]);
            let frontier = new Set([startId]);

            for (let hop = 0; hop < depth; hop++) {
                const next = new Set();
                for (const id of frontier) {
                    if (direction !== 'downstream') {
                        for (const up of varUp.get(id) || []) {
                            next.add(up);
                        }
                    }
                    if (direction !== 'upstream') {
                        for (const down of varDown.get(id) || []) {
                            next.add(down);
                        }
                    }
                }
                let added = false;
                for (const id of next) {
                    if (!scope.has(id)) {
                        scope.add(id);
                        added = true;
                    }
                }
                if (!added) {
                    break;
                }
                frontier = next;
            }
            return scope;
        }

        function selectedTraceEdges() {
            if (state.traceFilter) {
                if (state.traceFilter.kind === 'cell') {
                    const ids = new Set(state.traceFilter.edgeIds || []);
                    return edges.filter(function (edge) {
                        return ids.has(edge.id) && edgeAllowed(edge);
                    });
                }
                if (state.traceFilter.kind === 'path') {
                    const pairs = new Set();
                    const path = state.traceFilter.path || [];
                    for (let i = 0; i + 1 < path.length; i++) {
                        pairs.add(path[i] + '\u0000' + path[i + 1]);
                    }
                    return edges.filter(function (edge) {
                        return pairs.has(
                            edge.source_element + '\u0000' +
                            edge.target_element) && edgeAllowed(edge);
                    });
                }
            }

            if (!state.focus) {
                return [];
            }
            if (state.focus.type === 'variable') {
                const scope = variableScope(
                    state.focus.id, state.depth, state.direction);
                return edges.filter(function (edge) {
                    return scope.has(edge.variable) && edgeAllowed(edge);
                });
            }
            const scope = elementScope(
                state.focus.id, state.depth, state.direction);
            return edges.filter(function (edge) {
                return scope.has(edge.source_element) &&
                    scope.has(edge.target_element) &&
                    edgeAllowed(edge);
            });
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
            if (state.focus) {
                params.set(
                    'focus',
                    (state.focus.type === 'element' ? 'el:' : 'var:') +
                    state.focus.id);
            } else {
                params.delete('focus');
            }
            if (state.depth !== 1) {
                params.set('depth', String(state.depth));
            } else {
                params.delete('depth');
            }
            if (state.direction !== 'both') {
                params.set('direction', state.direction);
            } else {
                params.delete('direction');
            }
            if (!state.showVariables) {
                params.set('vars', '0');
            } else {
                params.delete('vars');
            }
            if (!state.allowBackward) {
                params.set('backward', '0');
            } else {
                params.delete('backward');
            }
            if (state.allowSelf) {
                params.set('self', '1');
            } else {
                params.delete('self');
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
                    '</span></th>' + cells + '</tr>';
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
                        '</th>' + headerCells +
                    '</tr></thead>' +
                    '<tbody>' + bodyRows + '</tbody>' +
                '</table>';

            applyMatrixFilter();
        }

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
            state.traceFilter = {
                kind: 'cell',
                edgeIds: cell.edgeIds.slice(),
                variableIds: Array.from(cell.variables),
                source: source,
                target: target,
                label: elementLabel(source) + ' \u2192 ' +
                    elementLabel(target),
            };
            state.focus = { type: 'element', id: source };
            state.pathResults = null;
            renderPathResults(null);
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
            '<div class="mcmc-trace-controls">' +
                '<label class="mcmc-field">' +
                    '<span>Focus</span>' +
                    '<select data-role="focus"></select>' +
                '</label>' +
                '<label class="mcmc-field">' +
                    '<span>Depth</span>' +
                    '<select data-role="depth">' +
                        '<option value="1">1 hop</option>' +
                        '<option value="2">2 hops</option>' +
                        '<option value="3">3 hops</option>' +
                    '</select>' +
                '</label>' +
                '<label class="mcmc-field">' +
                    '<span>Direction</span>' +
                    '<select data-role="direction">' +
                        '<option value="both">Both</option>' +
                        '<option value="upstream">Upstream</option>' +
                        '<option value="downstream">Downstream</option>' +
                    '</select>' +
                '</label>' +
                '<label class="mcmc-check">' +
                    '<input type="checkbox" data-role="showVariables" ' +
                        'checked> Variable nodes' +
                '</label>' +
                '<label class="mcmc-check">' +
                    '<input type="checkbox" data-role="allowBackward" ' +
                        'checked> Carry / backward links' +
                '</label>' +
                '<label class="mcmc-check">' +
                    '<input type="checkbox" data-role="allowSelf"> ' +
                    'Self dependencies' +
                '</label>' +
                '<button type="button" class="mcmc-button" ' +
                    'data-role="resetTrace">Reset</button>' +
            '</div>' +
            '<div class="mcmc-trace-filter" data-role="filter" hidden>' +
            '</div>' +
            '<div class="mcmc-path-controls">' +
                '<strong>Path finder</strong>' +
                '<label class="mcmc-field">' +
                    '<span>From</span>' +
                    '<select data-role="pathFrom"></select>' +
                '</label>' +
                '<span class="mcmc-path-arrow">\u2192</span>' +
                '<label class="mcmc-field">' +
                    '<span>To</span>' +
                    '<select data-role="pathTo"></select>' +
                '</label>' +
                '<label class="mcmc-check">' +
                    '<input type="checkbox" data-role="pathBackward" ' +
                        'checked> Allow carry links' +
                '</label>' +
                '<button type="button" class="mcmc-button ' +
                    'mcmc-button-primary" data-role="findPaths">' +
                    'Find paths</button>' +
            '</div>' +
            '<div class="mcmc-path-results" data-role="pathResults"></div>' +
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
                '<aside class="mcmc-trace-sidebar">' +
                    '<div class="mcmc-details" data-role="details">' +
                    '</div>' +
                '</aside>' +
            '</div>';

        const traceControls = tracePane.querySelector(
            '.mcmc-trace-controls');
        const filterBar = tracePane.querySelector('[data-role="filter"]');
        const pathResultsEl = tracePane.querySelector(
            '[data-role="pathResults"]');
        const graphEl = tracePane.querySelector('[data-role="graph"]');
        const detailsEl = tracePane.querySelector('[data-role="details"]');

        function option(value, label) {
            return '<option value="' + esc(value) + '">' +
                esc(label) + '</option>';
        }

        function populateTraceSelects() {
            const focusSelect = tracePane.querySelector(
                '[data-role="focus"]');
            const elementOptions = elements.map(function (element) {
                return option('el:' + element.id, elementLabel(element.id));
            }).join('');
            const variableOptions = variables.map(function (variable) {
                return option(
                    'var:' + variable.id, variableOptionLabel(variable));
            }).join('');
            focusSelect.innerHTML =
                '<optgroup label="Elements">' + elementOptions +
                '</optgroup>' +
                '<optgroup label="Variables">' + variableOptions +
                '</optgroup>';

            const pathFrom = tracePane.querySelector(
                '[data-role="pathFrom"]');
            const pathTo = tracePane.querySelector(
                '[data-role="pathTo"]');
            const pathOptions = elements.map(function (element) {
                return option(element.id, elementLabel(element.id));
            }).join('');
            pathFrom.innerHTML = pathOptions;
            pathTo.innerHTML = pathOptions;

            let fromId = state.focus && state.focus.type === 'element'
                ? state.focus.id : elements[0].id;
            let toId = null;
            for (const element of elements) {
                if (element.id !== fromId) {
                    toId = element.id;
                    break;
                }
            }
            if (toId === null) {
                toId = fromId;
            }
            state.pathFrom = fromId;
            state.pathTo = toId;
            pathFrom.value = fromId;
            pathTo.value = toId;
        }

        function syncTraceControls() {
            const focusSelect = tracePane.querySelector(
                '[data-role="focus"]');
            const depthSelect = tracePane.querySelector(
                '[data-role="depth"]');
            const directionSelect = tracePane.querySelector(
                '[data-role="direction"]');
            const showVariables = tracePane.querySelector(
                '[data-role="showVariables"]');
            const allowBackward = tracePane.querySelector(
                '[data-role="allowBackward"]');
            const allowSelf = tracePane.querySelector(
                '[data-role="allowSelf"]');
            const pathFrom = tracePane.querySelector(
                '[data-role="pathFrom"]');
            const pathTo = tracePane.querySelector(
                '[data-role="pathTo"]');

            if (state.focus) {
                focusSelect.value = (state.focus.type === 'element'
                    ? 'el:' : 'var:') + state.focus.id;
            }
            depthSelect.value = String(state.depth);
            directionSelect.value = state.direction;
            showVariables.checked = state.showVariables;
            allowBackward.checked = state.allowBackward;
            allowSelf.checked = state.allowSelf;

            if (state.pathFrom) {
                pathFrom.value = state.pathFrom;
            }
            if (state.pathTo) {
                pathTo.value = state.pathTo;
            }
            renderFilterBar();
        }

        function renderFilterBar() {
            if (!state.traceFilter) {
                filterBar.hidden = true;
                filterBar.innerHTML = '';
                return;
            }
            const kindLabel = state.traceFilter.kind === 'cell'
                ? 'Matrix cell' : 'Path';
            filterBar.hidden = false;
            filterBar.innerHTML =
                '<span><strong>' + esc(kindLabel) + ':</strong> ' +
                    esc(state.traceFilter.label) + '</span>' +
                '<button type="button" class="mcmc-button" ' +
                    'data-role="clearFilter">Clear filter</button>';
        }

        function setFocus(kind, id, options) {
            if (kind === 'element' && !elementsById.has(id)) {
                return;
            }
            if (kind === 'variable' && !variablesById.has(id)) {
                return;
            }
            state.focus = { type: kind, id: id };
            if (!options || !options.keepFilter) {
                state.traceFilter = null;
            }
            state.pathResults = null;
            renderPathResults(null);
            if (state.tab === 'trace') {
                syncTraceControls();
                renderTrace();
                updateUrl();
            } else {
                setTab('trace');
            }
        }

        function resetTrace() {
            state.focus = defaultFocus();
            state.depth = 1;
            state.direction = 'both';
            state.showVariables = true;
            state.allowBackward = true;
            state.allowSelf = false;
            state.traceFilter = null;
            state.pathResults = null;
            renderPathResults(null);
            syncTraceControls();
            renderTrace();
            updateUrl();
        }

        traceControls.addEventListener('change', function (event) {
            const role = event.target.dataset.role;
            if (role === 'focus') {
                const value = event.target.value;
                const colon = value.indexOf(':');
                setFocus(value.slice(0, colon), value.slice(colon + 1));
            } else if (role === 'depth') {
                state.depth = parseInt(event.target.value, 10) || 1;
                renderTrace();
                updateUrl();
            } else if (role === 'direction') {
                state.direction = event.target.value;
                renderTrace();
                updateUrl();
            } else if (role === 'showVariables') {
                state.showVariables = event.target.checked;
                renderTrace();
                updateUrl();
            } else if (role === 'allowBackward') {
                state.allowBackward = event.target.checked;
                renderTrace();
                updateUrl();
            } else if (role === 'allowSelf') {
                state.allowSelf = event.target.checked;
                renderTrace();
                updateUrl();
            }
        });

        tracePane.addEventListener('click', function (event) {
            const role = event.target.dataset.role;
            if (role === 'resetTrace') {
                resetTrace();
                return;
            }
            if (role === 'clearFilter') {
                state.traceFilter = null;
                renderTrace();
                updateUrl();
                return;
            }
            if (role === 'findPaths') {
                runPathFinder();
                return;
            }
            const pathButton = event.target.closest('[data-path-index]');
            if (pathButton) {
                selectPath(parseInt(pathButton.dataset.pathIndex, 10));
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
                    'text-wrap': 'wrap',
                    'text-max-width': 150,
                    'text-valign': 'center',
                    'text-halign': 'center',
                    'background-color': 'data(color)',
                    'color': '#ffffff',
                    'border-width': 1,
                    'border-color': '#ffffff',
                    'shape': 'round-rectangle',
                    'width': 'label',
                    'height': 'label',
                    'padding': '7px',
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
                selector: 'edge[kind="direct"]',
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

        function buildGraphModel(traceEdges) {
            const nodes = new Map();
            const graphEdges = [];

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

            function addOccurrenceNode(occurrenceId) {
                const occurrence = occurrencesById.get(occurrenceId);
                if (!occurrence) {
                    return null;
                }
                return addMethodNode(
                    occurrence.element, occurrence.method);
            }

            function addVariableNode(variableId) {
                const variable = variablesById.get(variableId);
                if (!variable) {
                    return null;
                }
                const id = 'v:' + variable.id;
                if (!nodes.has(id)) {
                    nodes.set(id, {
                        data: {
                            id: id,
                            type: 'variable',
                            variable: variable.id,
                            kind: variable.kind,
                            label: variableLabel(variable.id),
                            color: KIND_COLORS[variable.kind] ||
                                KIND_COLORS.general,
                        },
                    });
                }
                return id;
            }

            for (const edge of traceEdges) {
                const sourceNode = addOccurrenceNode(edge.source);
                const targetNode = addOccurrenceNode(edge.target);
                if (!sourceNode || !targetNode) {
                    continue;
                }
                if (state.showVariables) {
                    const variableNode = addVariableNode(edge.variable);
                    if (!variableNode) {
                        continue;
                    }
                    graphEdges.push({
                        data: {
                            id: 'e:' + edge.id + ':def',
                            source: sourceNode,
                            target: variableNode,
                            kind: 'define',
                            variable: edge.variable,
                            backward: false,
                        },
                    });
                    graphEdges.push({
                        data: {
                            id: 'e:' + edge.id + ':acc',
                            source: variableNode,
                            target: targetNode,
                            kind: 'access',
                            variable: edge.variable,
                            backward: !!edge.backward,
                        },
                    });
                } else {
                    graphEdges.push({
                        data: {
                            id: 'e:' + edge.id,
                            source: sourceNode,
                            target: targetNode,
                            kind: 'direct',
                            variable: edge.variable,
                            backward: !!edge.backward,
                            label: variableLabel(edge.variable),
                        },
                    });
                }
            }

            if (state.focus && state.focus.type === 'variable') {
                addVariableNode(state.focus.id);
            }
            if (state.focus && state.focus.type === 'element') {
                const element = elementsById.get(state.focus.id);
                if (element) {
                    for (const method of METHOD_ORDER) {
                        const methodData = element.methods &&
                            element.methods[method];
                        if (methodData &&
                                ((methodData.defines || []).length ||
                                 (methodData.accesses || []).length)) {
                            addMethodNode(element.id, method);
                        }
                    }
                }
            }

            return {
                nodes: Array.from(nodes.values()),
                edges: graphEdges,
            };
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
                    'No dependencies match the current focus and filters.' +
                    '</div>';
                return;
            }

            const elementsForCy = graphModel.nodes.concat(graphModel.edges);
            cy = window.cytoscape({
                container: graphEl,
                elements: elementsForCy,
                style: CY_STYLE,
                layout: {
                    name: 'breadthfirst',
                    directed: true,
                    padding: 24,
                    spacingFactor: 1.15,
                    nodeDimensionsIncludeLabels: true,
                    animate: false,
                    fit: true,
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
                    showVariableDetails(nodeData.variable);
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

        function elementAdjacency(allowBackward) {
            const adjacency = new Map();
            for (const edge of edges) {
                if (!allowBackward && edge.backward) {
                    continue;
                }
                if (edge.source_element === edge.target_element) {
                    continue;
                }
                if (!adjacency.has(edge.source_element)) {
                    adjacency.set(edge.source_element, new Set());
                }
                adjacency.get(edge.source_element).add(edge.target_element);
            }
            return adjacency;
        }

        function findDependencyPaths(fromId, toId, allowBackward) {
            if (fromId === toId) {
                return {
                    paths: [[fromId]],
                    distance: 0,
                    from: fromId,
                    to: toId,
                };
            }
            const adjacency = elementAdjacency(allowBackward);
            const distance = new Map();
            distance.set(fromId, 0);
            const queue = [fromId];
            while (queue.length) {
                const id = queue.shift();
                for (const next of adjacency.get(id) || []) {
                    if (!distance.has(next)) {
                        distance.set(next, distance.get(id) + 1);
                        queue.push(next);
                    }
                }
            }
            if (!distance.has(toId)) {
                return {
                    paths: [],
                    distance: null,
                    from: fromId,
                    to: toId,
                };
            }

            const maxLength = distance.get(toId) + 1;
            const paths = [];
            const stack = [fromId];
            const visited = new Set([fromId]);
            let steps = 0;

            function dfs(id) {
                if (paths.length >= MAX_PATHS ||
                        steps >= MAX_PATH_SEARCH_STEPS) {
                    return;
                }
                steps += 1;
                if (id === toId) {
                    paths.push(stack.slice());
                    return;
                }
                if (stack.length - 1 >= maxLength) {
                    return;
                }
                for (const next of adjacency.get(id) || []) {
                    if (visited.has(next)) {
                        continue;
                    }
                    visited.add(next);
                    stack.push(next);
                    dfs(next);
                    stack.pop();
                    visited.delete(next);
                }
            }

            dfs(fromId);
            paths.sort(function (a, b) {
                return a.length - b.length;
            });
            return {
                paths: paths.slice(0, MAX_PATHS),
                distance: distance.get(toId),
                from: fromId,
                to: toId,
            };
        }

        function renderPathResults(result) {
            if (!result) {
                pathResultsEl.innerHTML = '';
                return;
            }
            if (!result.paths.length) {
                pathResultsEl.innerHTML =
                    '<div class="mcmc-path-results-inner">' +
                    '<p class="mcmc-path-message">No directed path from ' +
                    '<strong>' + esc(elementLabel(result.from)) +
                    '</strong> to <strong>' +
                    esc(elementLabel(result.to)) + '</strong>. ' +
                    'Try enabling carry / backward links.</p>' +
                    '</div>';
                return;
            }
            const items = result.paths.map(function (path, index) {
                const label = path.map(function (elementId, pathIndex) {
                    return (pathIndex
                        ? '<span class="mcmc-path-arrow-char">\u2192</span> '
                        : '') + esc(elementLabel(elementId));
                }).join(' ');
                return '<li><button type="button" class="mcmc-path" ' +
                    'data-path-index="' + index + '">' + label +
                    '</button></li>';
            }).join('');
            pathResultsEl.innerHTML =
                '<div class="mcmc-path-results-inner">' +
                    '<h4>' + result.paths.length + ' path' +
                        (result.paths.length === 1 ? '' : 's') +
                        ' \u00b7 shortest ' + result.distance +
                        ' hop' + (result.distance === 1 ? '' : 's') +
                    '</h4>' +
                    '<ul class="mcmc-path-list">' + items + '</ul>' +
                '</div>';
        }

        function runPathFinder() {
            const pathFrom = tracePane.querySelector(
                '[data-role="pathFrom"]');
            const pathTo = tracePane.querySelector(
                '[data-role="pathTo"]');
            const pathBackward = tracePane.querySelector(
                '[data-role="pathBackward"]');
            const fromId = pathFrom.value;
            const toId = pathTo.value;
            state.pathFrom = fromId;
            state.pathTo = toId;
            state.pathResults = findDependencyPaths(
                fromId, toId, pathBackward.checked);
            renderPathResults(state.pathResults);
        }

        function selectPath(index) {
            const result = state.pathResults;
            if (!result || !result.paths[index]) {
                return;
            }
            const path = result.paths[index];
            state.traceFilter = {
                kind: 'path',
                path: path,
                label: path.map(elementLabel).join(' \u2192 '),
            };
            state.focus = { type: 'element', id: path[0] };
            syncTraceControls();
            renderTrace();
            updateUrl();
            graphEl.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }

        /* -----------------------------------------------------------------
         * Trace: rendering
         * ---------------------------------------------------------------- */

        function renderTrace() {
            const traceEdges = selectedTraceEdges();
            const graphModel = buildGraphModel(traceEdges);
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
