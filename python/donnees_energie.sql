-- ================================================================
--  FICHIER SQL - DONNEES D'ENTRAINEMENT
--  Arbre de Decision - Selection d'Energie Renouvelable
--  Base de donnees : energie_renouvelable.db
--  Auteur          : Arbre de Decision IA
-- ================================================================


-- ----------------------------------------------------------------
--  TABLE 1 : Donnees d'entrainement du modele
-- ----------------------------------------------------------------
CREATE TABLE IF NOT EXISTS donnees_entrainement (
    id                     INTEGER  PRIMARY KEY AUTOINCREMENT,
    irradiation_solaire    REAL     NOT NULL,   -- kWh/m2/an
    vitesse_vent           REAL     NOT NULL,   -- m/s
    disponibilite_eau      TEXT     NOT NULL,   -- Oui / Non
    disponibilite_biomasse TEXT     NOT NULL,   -- Faible / Moyen / Eleve
    disponibilite_terrain  TEXT     NOT NULL,   -- Petit / Moyen / Grand
    temperature_moyenne    REAL     NOT NULL,   -- degres Celsius
    energie_recommandee    TEXT     NOT NULL    -- etiquette cible (classe)
);


-- ----------------------------------------------------------------
--  TABLE 2 : Historique des predictions utilisateur
-- ----------------------------------------------------------------
CREATE TABLE IF NOT EXISTS predictions (
    id                     INTEGER  PRIMARY KEY AUTOINCREMENT,
    date_prediction        TEXT     DEFAULT (datetime('now','localtime')),
    nom_site               TEXT,
    irradiation_solaire    REAL,
    vitesse_vent           REAL,
    disponibilite_eau      TEXT,
    disponibilite_biomasse TEXT,
    disponibilite_terrain  TEXT,
    temperature_moyenne    REAL,
    energie_recommandee    TEXT,
    confiance_pct          REAL
);


-- ----------------------------------------------------------------
--  TABLE 3 : Catalogue des sources d'energie (reference)
-- ----------------------------------------------------------------
CREATE TABLE IF NOT EXISTS sources_energie (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    code        TEXT    NOT NULL UNIQUE,
    nom_fr      TEXT    NOT NULL,
    description TEXT,
    couleur_hex TEXT
);

INSERT OR IGNORE INTO sources_energie (code, nom_fr, description, couleur_hex) VALUES
    ('SOLAIRE',  'Energie Solaire PV',  'Conversion directe du rayonnement solaire en electricite via des panneaux photovoltaiques',    '#FFD700'),
    ('EOLIEN',   'Energie Eolienne',    'Conversion de l energie cinetique du vent en electricite via des eoliennes',                    '#87CEEB'),
    ('HYDRO',    'Hydroelectricite',    'Production d electricite par la force de l eau (barrages, cours d eau)',                        '#4682B4'),
    ('BIOMASSE', 'Energie Biomasse',    'Production d energie par combustion ou fermentation de matieres organiques',                    '#228B22');


-- ================================================================
--  DONNEES D'ENTRAINEMENT (40 echantillons - 4 classes x 10)
-- ================================================================

-- ----------------------------------------------------------------
--  CLASSE 1 : Energie Solaire PV
--  Conditions : forte irradiation, faible vent, pas d eau, terrain grand
-- ----------------------------------------------------------------
INSERT INTO donnees_entrainement
    (irradiation_solaire, vitesse_vent, disponibilite_eau, disponibilite_biomasse, disponibilite_terrain, temperature_moyenne, energie_recommandee)
VALUES
    (2200, 3.0, 'Non', 'Faible', 'Grand',  28, 'Energie Solaire PV'),
    (2100, 5.5, 'Non', 'Faible', 'Grand',  22, 'Energie Solaire PV'),
    (1900, 4.0, 'Non', 'Faible', 'Moyen',  25, 'Energie Solaire PV'),
    (2300, 2.5, 'Non', 'Moyen',  'Grand',  30, 'Energie Solaire PV'),
    (2000, 3.5, 'Non', 'Faible', 'Grand',  27, 'Energie Solaire PV'),
    (1800, 4.5, 'Non', 'Faible', 'Moyen',  24, 'Energie Solaire PV'),
    (2400, 3.0, 'Non', 'Faible', 'Grand',  32, 'Energie Solaire PV'),
    (2150, 5.0, 'Non', 'Moyen',  'Grand',  29, 'Energie Solaire PV'),
    (1950, 2.0, 'Non', 'Faible', 'Grand',  26, 'Energie Solaire PV'),
    (2050, 4.0, 'Non', 'Faible', 'Petit',  23, 'Energie Solaire PV');

-- ----------------------------------------------------------------
--  CLASSE 2 : Energie Eolienne
--  Conditions : vitesse vent elevee (>7 m/s), irradiation moderee
-- ----------------------------------------------------------------
INSERT INTO donnees_entrainement
    (irradiation_solaire, vitesse_vent, disponibilite_eau, disponibilite_biomasse, disponibilite_terrain, temperature_moyenne, energie_recommandee)
VALUES
    (1400,  8.0, 'Non', 'Faible', 'Grand',  15, 'Energie Eolienne'),
    (1200,  9.5, 'Non', 'Moyen',  'Grand',  12, 'Energie Eolienne'),
    (1100,  7.5, 'Non', 'Faible', 'Moyen',  10, 'Energie Eolienne'),
    (1300,  8.5, 'Non', 'Faible', 'Grand',  14, 'Energie Eolienne'),
    (1500,  7.0, 'Non', 'Faible', 'Grand',  16, 'Energie Eolienne'),
    (1000,  9.0, 'Non', 'Moyen',  'Grand',  11, 'Energie Eolienne'),
    (1600,  8.0, 'Non', 'Faible', 'Moyen',  13, 'Energie Eolienne'),
    (1250, 10.0, 'Non', 'Faible', 'Grand',   9, 'Energie Eolienne'),
    (1350,  7.5, 'Non', 'Faible', 'Grand',  17, 'Energie Eolienne'),
    (1450,  8.5, 'Non', 'Moyen',  'Moyen',  15, 'Energie Eolienne');

-- ----------------------------------------------------------------
--  CLASSE 3 : Hydroelectricite
--  Conditions : disponibilite eau = Oui (critere principal)
-- ----------------------------------------------------------------
INSERT INTO donnees_entrainement
    (irradiation_solaire, vitesse_vent, disponibilite_eau, disponibilite_biomasse, disponibilite_terrain, temperature_moyenne, energie_recommandee)
VALUES
    (1600, 4.0, 'Oui', 'Faible', 'Moyen',  18, 'Hydroelectricite'),
    (1400, 3.5, 'Oui', 'Moyen',  'Grand',  16, 'Hydroelectricite'),
    (1700, 5.0, 'Oui', 'Faible', 'Grand',  20, 'Hydroelectricite'),
    (1300, 4.5, 'Oui', 'Faible', 'Petit',  15, 'Hydroelectricite'),
    (1500, 3.0, 'Oui', 'Moyen',  'Moyen',  19, 'Hydroelectricite'),
    (1800, 4.0, 'Oui', 'Eleve',  'Grand',  21, 'Hydroelectricite'),
    (1200, 5.5, 'Oui', 'Faible', 'Moyen',  14, 'Hydroelectricite'),
    (1600, 3.0, 'Oui', 'Moyen',  'Petit',  17, 'Hydroelectricite'),
    (1900, 4.0, 'Oui', 'Faible', 'Grand',  22, 'Hydroelectricite'),
    (1100, 3.5, 'Oui', 'Eleve',  'Moyen',  13, 'Hydroelectricite');

-- ----------------------------------------------------------------
--  CLASSE 4 : Energie Biomasse
--  Conditions : biomasse elevee, irradiation/vent moderes
-- ----------------------------------------------------------------
INSERT INTO donnees_entrainement
    (irradiation_solaire, vitesse_vent, disponibilite_eau, disponibilite_biomasse, disponibilite_terrain, temperature_moyenne, energie_recommandee)
VALUES
    (1300, 3.0, 'Non', 'Eleve', 'Grand',  18, 'Energie Biomasse'),
    (1400, 4.0, 'Non', 'Eleve', 'Moyen',  20, 'Energie Biomasse'),
    (1200, 3.5, 'Non', 'Eleve', 'Grand',  16, 'Energie Biomasse'),
    (1500, 4.5, 'Non', 'Eleve', 'Petit',  22, 'Energie Biomasse'),
    (1100, 2.5, 'Non', 'Eleve', 'Moyen',  15, 'Energie Biomasse'),
    (1600, 3.0, 'Non', 'Eleve', 'Grand',  19, 'Energie Biomasse'),
    (1000, 4.0, 'Non', 'Eleve', 'Moyen',  14, 'Energie Biomasse'),
    (1700, 3.5, 'Non', 'Eleve', 'Grand',  21, 'Energie Biomasse'),
    (1300, 5.0, 'Non', 'Moyen', 'Petit',  17, 'Energie Biomasse'),
    (1200, 3.0, 'Non', 'Eleve', 'Petit',  16, 'Energie Biomasse');


-- ================================================================
--  REQUETES UTILES (pour consultation / verification)
-- ================================================================

-- Compter les echantillons par classe :
-- SELECT energie_recommandee, COUNT(*) AS total
-- FROM donnees_entrainement
-- GROUP BY energie_recommandee
-- ORDER BY total DESC;

-- Voir les statistiques par classe :
-- SELECT energie_recommandee,
--        ROUND(AVG(irradiation_solaire),1) AS moy_solaire,
--        ROUND(AVG(vitesse_vent),2)        AS moy_vent,
--        ROUND(AVG(temperature_moyenne),1) AS moy_temp
-- FROM donnees_entrainement
-- GROUP BY energie_recommandee;

-- Voir l'historique des predictions :
-- SELECT * FROM predictions ORDER BY id DESC;
