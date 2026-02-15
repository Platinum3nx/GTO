const RANKS = "AKQJT98765432";
let HISTORY = [];
let CURRENT_GRID_DATA = {};

document.addEventListener("DOMContentLoaded", () => {
    renderGrid();

    document.getElementById("solve-btn").addEventListener("click", updateStrategy);

    // Initial fetch
    updateStrategy();
});

function renderGrid() {
    const table = document.getElementById("strategy-grid");
    table.innerHTML = "";

    for (let i = 0; i < 13; i++) {
        const row = document.createElement("tr");
        for (let j = 0; j < 13; j++) {
            const cell = document.createElement("td");
            const r1 = RANKS[i];
            const r2 = RANKS[j];
            let hand = "";

            if (i === j) {
                // Pair
                hand = r1 + r2;
                cell.classList.add("pair");
            } else if (i < j) {
                // Suited (Upper Right)
                hand = r1 + r2 + "s";
                cell.classList.add("suited");
            } else {
                // Offsuit (Lower Left)
                hand = r2 + r1 + "o"; // Note: Offsuit usually written HighLow "AKo"
                cell.classList.add("offsuit");
            }

            cell.textContent = hand;
            cell.id = `cell-${hand}`;
            cell.onclick = () => selectHand(hand);
            row.appendChild(cell);
        }
        table.appendChild(row);
    }
}

async function updateStrategy() {
    const btn = document.getElementById("solve-btn");
    btn.disabled = true;
    btn.textContent = "Solving...";

    const board = document.getElementById("board-input").value;
    const pot = parseFloat(document.getElementById("pot-input").value);
    const stack = parseFloat(document.getElementById("stack-input").value);

    try {
        const response = await fetch("/api/strategy", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                board: board,
                pot: pot,
                stack: stack,
                history: HISTORY
            })
        });

        if (!response.ok) {
            const err = await response.json();
            alert("Error: " + err.detail);
            return;
        }

        const data = await response.json();
        CURRENT_GRID_DATA = data.grid || {};

        applyGridColors(CURRENT_GRID_DATA);

        // Select 'AA' by default if nothing selected
        selectHand("AA");

    } catch (e) {
        console.error(e);
        alert("Failed to fetch strategy.");
    } finally {
        btn.disabled = false;
        btn.textContent = "Update Strategy";
    }
}

function applyGridColors(gridData) {
    if (!gridData) return;

    for (const [hand, strat] of Object.entries(gridData)) {
        const cell = document.getElementById(`cell-${hand}`);
        if (!cell) continue;

        // Find dominant action
        let maxProb = 0;
        let maxAction = "CHECK";

        // Sum up aggression
        let aggression = (strat["BET_33"] || 0) + (strat["BET_75"] || 0) + (strat["ALL_IN"] || 0);
        let check = (strat["CHECK"] || 0) + (strat["CHECK/CALL"] || 0) + (strat["CALL"] || 0);
        let fold = strat["FOLD"] || 0;

        // Simple RGB mixing based on Check vs Bet vs Fold?
        // Let's do dominant strategy color for clarity + brightness for frequency
        // Green: Check, Red: Bet, Brown: Fold

        let color = "";

        if (fold > 0.5) {
            color = `rgba(93, 64, 55, ${fold})`; // Brown
        } else if (aggression > check) {
            color = `rgba(255, 51, 102, ${Math.min(aggression + 0.2, 1)})`; // Red
        } else {
            color = `rgba(0, 230, 118, ${Math.min(check + 0.2, 1)})`; // Green
        }

        cell.style.backgroundColor = color;
        // Contrast text
        cell.style.color = (fold > 0.7 || aggression > 0.7) ? "white" : "#eee";
    }
}

function selectHand(hand) {
    const display = document.getElementById("selected-hand-display");
    display.textContent = hand;

    const strat = CURRENT_GRID_DATA[hand];
    const barsContainer = document.getElementById("strategy-bars");
    barsContainer.innerHTML = "";

    if (!strat) {
        barsContainer.textContent = "No data for this hand.";
        return;
    }

    // Sort actions by logical order? FOLD, CHECK, BETs...
    const order = ["FOLD", "CHECK", "CHECK/CALL", "CALL", "BET_33", "BET_75", "ALL_IN"];

    for (const act of order) {
        if (strat[act] !== undefined) {
            const prob = strat[act];
            const pct = (prob * 100).toFixed(1) + "%";
            const row = document.createElement("div");
            row.className = "bar-item";
            row.innerHTML = `
                <div class="bar-label">${act}</div>
                <div class="bar-fill-container">
                    <div class="bar-fill" style="width: ${pct}; background-color: ${getActionColor(act)}"></div>
                </div>
                <div class="bar-value">${pct}</div>
             `;
            barsContainer.appendChild(row);
        }
    }
}

function getActionColor(action) {
    if (action.includes("FOLD")) return "#5d4037";
    if (action.includes("CHECK") || action.includes("CALL")) return "#00e676";
    if (action.includes("BET") || action.includes("ALL")) return "#ff3366";
    return "#aaa";
}

// History Management
function addAction(action) {
    HISTORY.push(action);
    renderHistory();
}

function resetHistory() {
    HISTORY = [];
    renderHistory();
}

function renderHistory() {
    const list = document.getElementById("history-list");
    list.innerHTML = HISTORY.map((act, idx) => `
        <div class="history-item">
            <span class="step-num">${idx + 1}.</span> ${act}
        </div>
    `).join("");
}
