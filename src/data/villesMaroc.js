// Donnees reelles des villes du Maroc
// Sources: NASA POWER, IRENA, ONM Maroc

const VILLES_MAROC = [
  // ── Grand Sud - Zone solaire premium ──────────────────────────
  { nom: "Guelmim",        region: "Guelmim-Oued Noun",          solaire: 2100, vent: 5.5, eau: "Non", biomasse: "Faible", terrain: "Grand",  temperature: 22 },
  { nom: "Laayoune",       region: "Laayoune-Sakia El Hamra",    solaire: 2200, vent: 6.5, eau: "Non", biomasse: "Faible", terrain: "Grand",  temperature: 23 },
  { nom: "Dakhla",         region: "Dakhla-Oued Ed-Dahab",       solaire: 2250, vent: 8.5, eau: "Non", biomasse: "Faible", terrain: "Grand",  temperature: 22 },
  { nom: "Tan-Tan",        region: "Guelmim-Oued Noun",          solaire: 2150, vent: 7.0, eau: "Non", biomasse: "Faible", terrain: "Grand",  temperature: 22 },
  { nom: "Smara",          region: "Laayoune-Sakia El Hamra",    solaire: 2180, vent: 5.8, eau: "Non", biomasse: "Faible", terrain: "Grand",  temperature: 24 },
  { nom: "Boujdour",       region: "Laayoune-Sakia El Hamra",    solaire: 2230, vent: 7.5, eau: "Non", biomasse: "Faible", terrain: "Grand",  temperature: 21 },
  { nom: "Assa",           region: "Guelmim-Oued Noun",          solaire: 2160, vent: 5.2, eau: "Non", biomasse: "Faible", terrain: "Grand",  temperature: 25 },
  { nom: "Tata",           region: "Souss-Massa",                solaire: 2120, vent: 4.5, eau: "Non", biomasse: "Faible", terrain: "Grand",  temperature: 27 },
  { nom: "Zagora",         region: "Draa-Tafilalet",             solaire: 2200, vent: 4.0, eau: "Non", biomasse: "Faible", terrain: "Grand",  temperature: 29 },
  { nom: "Ouarzazate",     region: "Draa-Tafilalet",             solaire: 2150, vent: 3.5, eau: "Oui", biomasse: "Faible", terrain: "Grand",  temperature: 22 },
  { nom: "Errachidia",     region: "Draa-Tafilalet",             solaire: 2100, vent: 4.0, eau: "Oui", biomasse: "Faible", terrain: "Grand",  temperature: 23 },
  { nom: "Merzouga",       region: "Draa-Tafilalet",             solaire: 2220, vent: 3.8, eau: "Non", biomasse: "Faible", terrain: "Grand",  temperature: 28 },
  { nom: "Tinghir",        region: "Draa-Tafilalet",             solaire: 2080, vent: 3.8, eau: "Oui", biomasse: "Faible", terrain: "Moyen",  temperature: 22 },
  { nom: "Midelt",         region: "Draa-Tafilalet",             solaire: 1950, vent: 4.0, eau: "Oui", biomasse: "Faible", terrain: "Moyen",  temperature: 14 },

  // ── Centre et Plateaux ────────────────────────────────────────
  { nom: "Marrakech",      region: "Marrakech-Safi",             solaire: 1900, vent: 3.5, eau: "Non", biomasse: "Moyen",  terrain: "Grand",  temperature: 25 },
  { nom: "Agadir",         region: "Souss-Massa",                solaire: 1850, vent: 5.0, eau: "Non", biomasse: "Moyen",  terrain: "Moyen",  temperature: 20 },
  { nom: "Tiznit",         region: "Souss-Massa",                solaire: 1880, vent: 5.5, eau: "Non", biomasse: "Faible", terrain: "Moyen",  temperature: 21 },
  { nom: "Taroudant",      region: "Souss-Massa",                solaire: 1920, vent: 3.8, eau: "Oui", biomasse: "Moyen",  terrain: "Grand",  temperature: 24 },
  { nom: "Beni Mellal",    region: "Beni Mellal-Khenifra",       solaire: 1750, vent: 3.2, eau: "Oui", biomasse: "Eleve",  terrain: "Moyen",  temperature: 19 },
  { nom: "Khouribga",      region: "Beni Mellal-Khenifra",       solaire: 1700, vent: 3.8, eau: "Non", biomasse: "Moyen",  terrain: "Grand",  temperature: 18 },
  { nom: "Khenifra",       region: "Beni Mellal-Khenifra",       solaire: 1780, vent: 3.0, eau: "Oui", biomasse: "Eleve",  terrain: "Moyen",  temperature: 14 },
  { nom: "Settat",         region: "Casablanca-Settat",          solaire: 1680, vent: 4.0, eau: "Non", biomasse: "Moyen",  terrain: "Grand",  temperature: 18 },
  { nom: "El Jadida",      region: "Casablanca-Settat",          solaire: 1650, vent: 5.5, eau: "Non", biomasse: "Moyen",  terrain: "Moyen",  temperature: 18 },
  { nom: "Safi",           region: "Marrakech-Safi",             solaire: 1700, vent: 6.0, eau: "Non", biomasse: "Faible", terrain: "Moyen",  temperature: 18 },
  { nom: "Essaouira",      region: "Marrakech-Safi",             solaire: 1680, vent: 8.5, eau: "Non", biomasse: "Faible", terrain: "Moyen",  temperature: 17 },
  { nom: "Casablanca",     region: "Casablanca-Settat",          solaire: 1620, vent: 4.5, eau: "Non", biomasse: "Faible", terrain: "Petit",  temperature: 18 },

  // ── Nord-Ouest et Atlantique ──────────────────────────────────
  { nom: "Rabat",          region: "Rabat-Sale-Kenitra",         solaire: 1600, vent: 4.5, eau: "Non", biomasse: "Moyen",  terrain: "Petit",  temperature: 18 },
  { nom: "Sale",           region: "Rabat-Sale-Kenitra",         solaire: 1590, vent: 4.8, eau: "Non", biomasse: "Faible", terrain: "Petit",  temperature: 18 },
  { nom: "Kenitra",        region: "Rabat-Sale-Kenitra",         solaire: 1580, vent: 4.5, eau: "Oui", biomasse: "Moyen",  terrain: "Moyen",  temperature: 18 },
  { nom: "Meknes",         region: "Fes-Meknes",                 solaire: 1700, vent: 3.5, eau: "Oui", biomasse: "Eleve",  terrain: "Moyen",  temperature: 17 },
  { nom: "Fes",            region: "Fes-Meknes",                 solaire: 1680, vent: 3.2, eau: "Oui", biomasse: "Eleve",  terrain: "Moyen",  temperature: 17 },
  { nom: "Ifrane",         region: "Fes-Meknes",                 solaire: 1750, vent: 3.0, eau: "Oui", biomasse: "Eleve",  terrain: "Moyen",  temperature:  8 },
  { nom: "Azrou",          region: "Fes-Meknes",                 solaire: 1720, vent: 3.2, eau: "Oui", biomasse: "Eleve",  terrain: "Moyen",  temperature: 10 },

  // ── Nord et Mediterranee ──────────────────────────────────────
  { nom: "Tanger",         region: "Tanger-Tetouan-Al Hoceima",  solaire: 1550, vent: 8.0, eau: "Non", biomasse: "Moyen",  terrain: "Petit",  temperature: 17 },
  { nom: "Tetouan",        region: "Tanger-Tetouan-Al Hoceima",  solaire: 1580, vent: 5.5, eau: "Oui", biomasse: "Moyen",  terrain: "Petit",  temperature: 17 },
  { nom: "Al Hoceima",     region: "Tanger-Tetouan-Al Hoceima",  solaire: 1650, vent: 5.0, eau: "Oui", biomasse: "Moyen",  terrain: "Petit",  temperature: 19 },
  { nom: "Chefchaouen",    region: "Tanger-Tetouan-Al Hoceima",  solaire: 1600, vent: 3.5, eau: "Oui", biomasse: "Eleve",  terrain: "Petit",  temperature: 16 },
  { nom: "Larache",        region: "Tanger-Tetouan-Al Hoceima",  solaire: 1530, vent: 5.5, eau: "Oui", biomasse: "Moyen",  terrain: "Moyen",  temperature: 17 },

  // ── Oriental ──────────────────────────────────────────────────
  { nom: "Oujda",          region: "Oriental",                   solaire: 1800, vent: 4.5, eau: "Non", biomasse: "Moyen",  terrain: "Grand",  temperature: 19 },
  { nom: "Nador",          region: "Oriental",                   solaire: 1750, vent: 5.0, eau: "Non", biomasse: "Moyen",  terrain: "Moyen",  temperature: 19 },
  { nom: "Berkane",        region: "Oriental",                   solaire: 1780, vent: 4.8, eau: "Oui", biomasse: "Moyen",  terrain: "Moyen",  temperature: 19 },
  { nom: "Taourirt",       region: "Oriental",                   solaire: 1820, vent: 5.0, eau: "Non", biomasse: "Faible", terrain: "Grand",  temperature: 20 },
  { nom: "Figuig",         region: "Oriental",                   solaire: 2050, vent: 4.2, eau: "Non", biomasse: "Faible", terrain: "Grand",  temperature: 26 },
];

export default VILLES_MAROC;
