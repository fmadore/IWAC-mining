// Configuration constants
window.CONFIG = {
    dataUrl: 'https://github.com/fmadore/Mining_IWAC/raw/refs/heads/main/Integrisme/Word_cloud/data/word_frequencies.json',
    width: 800,
    height: 600,
    padding: 5,
    minFontSize: 10,
    maxFontSize: 60
};

window.WORD_CATEGORIES = {
    religious: ['religieux', 'musulman', 'islam', 'islamique', 'religion', 'imam', 'mosquée', 'prophète', 'coran', 'dieu', 'foi'],
    political: ['politique', 'président', 'ministre', 'gouvernement', 'état', 'national', 'parti'],
    social: ['dialogue', 'communauté', 'association', 'social', 'culture'],
    places: ['ivoire', 'burkina', 'pays', 'monde', 'faso', 'bénin']
};

window.CATEGORY_COLORS = {
    religious: d3.interpolateReds,
    political: d3.interpolateBlues,
    social: d3.interpolateGreens,
    places: d3.interpolateOranges,
    default: d3.interpolatePurples
}; 