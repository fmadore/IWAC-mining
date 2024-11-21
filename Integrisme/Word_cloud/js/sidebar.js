// Sidebar management
class Sidebar {
    constructor() {
        this.container = document.getElementById('top-words-list');
    }

    update(words) {
        const topWords = words.slice(0, 25);
        const html = `
            <ul class="word-list">
                ${topWords.map((word, index) => `
                    <li class="word-list-item">
                        <span class="word-rank">${index + 1}.</span>
                        <span class="word-text">${word.text}</span>
                        <span class="word-frequency">${Utils.formatNumber(word.frequency)}</span>
                    </li>
                `).join('')}
            </ul>
        `;
        
        this.container.innerHTML = html;
    }
} 