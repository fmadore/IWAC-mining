const CONFIG = {
    dataUrl: './data/word_frequencies.json',
    wordCloud: {
        width: 800,
        height: 600,
        padding: 5,
        rotationAngles: [-60, -30, 0, 30, 60],
        fontSizeScale: [12, 80],
        fontFamily: 'Arial'
    },
    colors: {
        default: d3.scaleLinear()
            .domain([0, 1])
            .range(['#4a90e2', '#357abd']),
        religious: d3.scaleLinear()
            .domain([0, 1])
            .range(['#e24a4a', '#bd3535']),
        political: d3.scaleLinear()
            .domain([0, 1])
            .range(['#4ae24a', '#35bd35'])
    }
};

const WORD_CATEGORIES = {
    religious: ['musulman', 'religieux', 'islam', 'islamique', 'religion', 'dieu', 'imam', 'mosquée', 'coran', 'prophète', 'foi', 'allah'],
    political: ['politique', 'président', 'pouvoir', 'état', 'gouvernement', 'ministre', 'parti']
};

const CATEGORY_COLORS = CONFIG.colors; 