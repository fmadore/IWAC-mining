const MapConfig = {
    width: 960,
    height: 600,
    projection: {
        scale: 150,
        center: [0, 20],
        translate: [480, 300]
    },
    colors: {
        background: "#f0f0f0",
        land: "#d3d3d3",
        stroke: "#999",
        circles: "red",
        choropleth: {
            noData: "#f0f0f0",
            scale: [
                "#fee5d9",  // 1-4
                "#fcbba1",  // 5-9
                "#fc9272",  // 10-24
                "#fb6a4a",  // 25-49
                "#ef3b2c",  // 50-99
                "#cb181d",  // 100-149
                "#99000d",  // 150+
                "#67000d"   // Adding an extra darker color for 150+
            ]
        }
    },
    zoom: {
        min: 1,
        max: 8
    },
    circle: {
        minRadius: 5,
        maxRadius: 25,
        minOpacity: 0.5,
        maxOpacity: 0.8
    },
    countryNameMapping: {
        "Algérie": "Algeria",
        "France": "France",
        "Burkina Faso": "Burkina Faso",
        "Mali": "Mali",
        "Côte d'Ivoire": "Ivory Coast",
        "Bénin": "Benin",
        "Togo": "Togo",
        "Ghana": "Ghana",
        "Nigéria": "Nigeria",
        "Niger": "Niger",
        "Tchad": "Chad",
        "Libye": "Libya",
        "Tunisie": "Tunisia",
        "Maroc": "Morocco",
        "Mauritanie": "Mauritania",
        "Sénégal": "Senegal",
        "Gambie": "Gambia",
        "Guinée": "Guinea",
        "Sierra Leone": "Sierra Leone",
        "Liberia": "Liberia",
        "Cameroun": "Cameroon",
        "République centrafricaine": "Central African Republic",
        "Soudan": "Sudan",
        "Égypte": "Egypt",
        "Arabie saoudite": "Saudi Arabia",
        "Iran": "Iran",
        "Irak": "Iraq",
        "Syrie": "Syria",
        "Turquie": "Turkey",
        "Liban": "Lebanon",
        "Israël": "Israel",
        "Palestine": "Palestine",
        "Koweït": "Kuwait",
        "Qatar": "Qatar",
        "Émirats arabes unis": "United Arab Emirates",
        "Afghanistan": "Afghanistan",
        "Pakistan": "Pakistan",
        "Inde": "India",
        "Bangladesh": "Bangladesh",
        "Chine": "China",
        "États-Unis": "United States",
        "Canada": "Canada",
        "Royaume-Uni": "United Kingdom",
        "Angleterre": "England",
        "Irlande du Nord": "Northern Ireland",
        "Allemagne": "Germany",
        "Belgique": "Belgium",
        "Suisse": "Switzerland",
        "Espagne": "Spain",
        "Suède": "Sweden",
        "Bosnie-Herzégovine": "Bosnia and Herzegovina",
        "Tchétchénie": "Chechnya",
        "Somalie": "Somalia",
        "Rwanda": "Rwanda",
        "Burundi": "Burundi",
        "République du Congo": "Republic of Congo",
        "Île Maurice": "Mauritius",
        "Afrique du Sud": "South Africa"
    }
};

export default MapConfig; 