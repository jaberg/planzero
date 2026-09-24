// Per-table sort state: WeakMap keyed by table element
const tableStates = new WeakMap();

/**
 * Returns the sort state object for the given table, creating it on first use.
 * @param {HTMLTableElement} table - The table element.
 * @returns {{columnIndex: number, direction: string, initialWidthsSet: boolean}}
 */
function getTableState(table) {
    let state = tableStates.get(table);
    if (!state) {
        state = {
            columnIndex: -1,
            direction: 'asc', // 'asc' for ascending, 'desc' for descending
            initialWidthsSet: false
        };
        tableStates.set(table, state);
    }
    return state;
}

/**
 * Sorts the HTML table rows based on the content of a specific column.
 * @param {HTMLTableElement} table - The table element to sort.
 * @param {number} columnIndex - The zero-based index of the column to sort.
 */
function sortTable(table, columnIndex) {
    const state = getTableState(table);
    const tbody = table.querySelector('tbody');
    const rows = Array.from(tbody.querySelectorAll('tr'));
    const headers = table.querySelectorAll('th');

    // 1. Determine new sort direction
    let newDirection = 'asc';
    if (state.columnIndex === columnIndex) {
        // Toggle direction if the same column is clicked
        newDirection = state.direction === 'asc' ? 'desc' : 'asc';
    }

    // 2. Sorting Logic
    const sortedRows = rows.sort((a, b) => {
        // Get the cell content for comparison
        const aText = a.children[columnIndex].textContent.trim();
        const bText = b.children[columnIndex].textContent.trim();

        // Check if the values can be treated as numbers
        const aNum = parseFloat(aText.replace(/[^0-9.-]+/g,"")); // Clean non-numeric characters for comparison
        const bNum = parseFloat(bText.replace(/[^0-9.-]+/g,""));

        let comparison = 0;

        if (!isNaN(aNum) && !isNaN(bNum) && isFinite(aNum) && isFinite(bNum)) {
            // Numerical comparison
            comparison = bNum - aNum;
        } else {
            // String (lexicographical) comparison
            comparison = aText.localeCompare(bText);
        }

        // Apply the direction
        return newDirection === 'asc' ? comparison : comparison * -1;
    });

    // 3. Update the table body with sorted rows
    tbody.innerHTML = ''; // Clear existing rows
    sortedRows.forEach(row => tbody.appendChild(row));

    // 4. Update UI (Header Indicators)
    // Remove indicators from all headers
    headers.forEach((th) => {
        th.classList.remove('th-sorted-asc', 'th-sorted-desc', 'bg-indigo-100');
        let indicator = th.querySelector('.sort-indicator');
        if (indicator) {
            indicator.remove();
        }
    });

    // Add indicator to the currently sorted header
    const currentHeader = headers[columnIndex];
    currentHeader.classList.add(`th-sorted-${newDirection}`, 'bg-indigo-100');
    
    // Add the arrow icon for the indicator
    const indicatorIcon = document.createElement('span');
    indicatorIcon.className = 'sort-indicator';
    indicatorIcon.innerHTML = '&#9660;'; // Down arrow
    currentHeader.appendChild(indicatorIcon);

    // 5. Update the table state
    state.columnIndex = columnIndex;
    state.direction = newDirection;
}

/**
 * Captures the initial, automatically calculated column widths and applies them as fixed styles.
 * This function must run BEFORE we set the table to fixed layout, or it won't work correctly.
 * @param {HTMLTableElement} table - The table element to fix widths for.
 */
function fixColumnWidths(table) {
    const state = getTableState(table);
    if (state.initialWidthsSet) return;

    const headers = table.querySelectorAll('th');

    // 1. Temporarily ensure the table-layout is AUTO so the browser calculates optimal widths
    // This is often the default, but we ensure it before calculation.
    table.style.tableLayout = 'auto';

    // 2. Capture the calculated width for each header cell
    headers.forEach((header) => {
        // Get the computed width in pixels
        const width = header.offsetWidth + 'px';
        // Apply the captured width as an inline style
        header.style.width = width;
    });

    // 3. Set the table layout to FIXED immediately after capturing and setting widths.
    // This locks the structure, preventing reflow during sorting.
    table.style.tableLayout = 'fixed';

    state.initialWidthsSet = true;
}


/**
 * Attaches the click listener to all sortable headers when the DOM is ready.
 */
document.addEventListener('DOMContentLoaded', () => {
    document.querySelectorAll('table.gemini-sort-data-table').forEach((table) => {
        const headers = table.querySelectorAll('th');

        // 1. Capture and fix column widths based on initial content
        // The browser must render the table before we can accurately measure widths.
        fixColumnWidths(table);

        headers.forEach((header, index) => {
            // Add a class for visual styling
            header.classList.add('sortable-th');
            
            // Attach the sorting function to the click event
            header.addEventListener('click', () => {
                sortTable(table, index);
            });
        });

        // Optional: Sort by the first column initially (Name)
        sortTable(table, 0);
    });
});