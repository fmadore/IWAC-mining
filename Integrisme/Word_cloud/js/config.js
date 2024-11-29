// Configuration constants
const CONFIG = {
    width: 1000,
    height: 600,
    minFontSize: 12,
    maxFontSize: 80,
    transitionDuration: 200
};

const WORD_CATEGORIES = {
    religious: ['religieux', 'musulman', 'islam', 'islamique', 'religion', 'imam', 'mosquée', 'prophète', 'coran', 'dieu', 'foi'],
    political: ['politique', 'président', 'ministre', 'gouvernement', 'état', 'national', 'parti'],
    social: ['dialogue', 'communauté', 'association', 'social', 'culture'],
    places: ['ivoire', 'burkina', 'pays', 'monde', 'faso', 'bénin']
};

const CATEGORY_COLORS = {
    religious: d3.interpolateReds,
    political: d3.interpolateBlues,
    social: d3.interpolateGreens,
    places: d3.interpolateOranges,
    default: d3.interpolatePurples
}; 