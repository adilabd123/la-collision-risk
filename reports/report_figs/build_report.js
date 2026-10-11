// Builds the project report as a Word document, mirroring the structure and
// formatting of the user's example (US Letter, Times New Roman 12 pt, blue
// headings, grid tables). Numbers come from data.json (generated from the
// project's outputs) so the text and tables never drift from the results.
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell,
  AlignmentType, HeadingLevel, WidthType, BorderStyle, LevelFormat, PageBreak,
  Footer, PageNumber, ShadingType, TabStopType,
} = require("docx");

const D = JSON.parse(fs.readFileSync(path.join(__dirname, "data.json"), "utf8"));
const REP = "/home/claude/la-collision-risk/reports";
const OUT = process.argv[2] || path.join(__dirname, "LA_Collision_Insurance_Risk_Report.docx");

const FONT = "Times New Roman";
const HFONT = "Cambria";
const CONTENT_W = 8640;            // 12240 - 2 × 1800 DXA side margins (as in the example)

// ------------------------------------------------------------------ helpers
// Inline **bold** and _italic_ markup → TextRuns
function runs(text, base = {}) {
  const out = [];
  const re = /(\*\*[^*]+\*\*|\{i\}[\s\S]+?\{\/i\})/g;
  let last = 0, m;
  while ((m = re.exec(text))) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), ...base }));
    const t = m[0];
    if (t.startsWith("**")) out.push(new TextRun({ text: t.slice(2, -2), bold: true, ...base }));
    else out.push(new TextRun({ text: t.slice(3, -4), italics: true, ...base }));
    last = m.index + t.length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), ...base }));
  return out;
}
const P = (text, opts = {}) => new Paragraph({
  children: runs(text), alignment: AlignmentType.JUSTIFIED, spacing: { after: 160, line: 300 }, ...opts,
});
const H1 = (text) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(text)] });
const H2 = (text) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(text)] });
const BUL = (text) => new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: runs(text),
  spacing: { after: 80, line: 276 } });
const NUM = (text, ref = "numbers") => new Paragraph({ numbering: { reference: ref, level: 0 },
  children: runs(text), spacing: { after: 80, line: 276 } });
const PLAIN = (text, opts = {}) => new Paragraph({ children: runs(text), spacing: { after: 60 }, ...opts });
const BREAK = () => new Paragraph({ children: [new PageBreak()] });
const CODE = (lines) => lines.map((l, i) => new Paragraph({
  children: [new TextRun({ text: l || " ", font: "Courier New", size: 17 })],
  spacing: { after: 0, line: 240 },
  shading: { type: ShadingType.CLEAR, fill: "F2F2F2", color: "auto" },
  indent: { left: 200, right: 200 },
}));

// Figures: numbered in order of appearance; caption below
let figNo = 0;
const FIGS = [];
function FIG(key, caption, widthIn = 5.6) {
  const [file, w, h] = D.images[key];
  figNo += 1;
  FIGS.push(caption);
  const wpx = Math.round(widthIn * 96), hpx = Math.round(wpx * h / w);
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 60 },
      keepNext: true,
      children: [new ImageRun({ type: "png", data: fs.readFileSync(path.join(REP, file)),
        transformation: { width: wpx, height: hpx },
        altText: { title: caption, description: caption, name: key } })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 240 },
      children: [new TextRun({ text: `Figure ${figNo}. ${caption}`, italics: true, size: 20 })] }),
  ];
}

// Tables: numbered in order of appearance; caption above; header row bold
let tabNo = 0;
const TABS = [];
const border = { style: BorderStyle.SINGLE, size: 4, color: "000000" };
const borders = { top: border, bottom: border, left: border, right: border };
function TAB(caption, header, rows, widths, align = null, size = 20) {
  tabNo += 1;
  TABS.push(caption);
  const total = widths.reduce((a, b) => a + b, 0);
  const scale = CONTENT_W / total;
  const cw = widths.map((w) => Math.round(w * scale));
  cw[cw.length - 1] += CONTENT_W - cw.reduce((a, b) => a + b, 0);
  const cell = (text, i, isHead) => new TableCell({
    borders, width: { size: cw[i], type: WidthType.DXA },
    margins: { top: 60, bottom: 60, left: 100, right: 100 },
    shading: isHead ? { type: ShadingType.CLEAR, fill: "DCE6F1", color: "auto" } : undefined,
    children: [new Paragraph({
      alignment: isHead ? AlignmentType.LEFT : ((align && align[i]) || AlignmentType.LEFT),
      spacing: { after: 0 },
      children: runs(String(text), { size, bold: isHead ? true : undefined }),
    })],
  });
  return [
    new Paragraph({ spacing: { before: 200, after: 80 }, keepNext: true,
      children: [new TextRun({ text: `Table ${tabNo}. ${caption}`, bold: true, size: 20 })] }),
    new Table({
      width: { size: CONTENT_W, type: WidthType.DXA }, columnWidths: cw,
      rows: [
        new TableRow({ tableHeader: true, children: header.map((t, i) => cell(t, i, true)) }),
        ...rows.map((r) => new TableRow({ cantSplit: true, children: r.map((t, i) => cell(t, i, false)) })),
      ],
    }),
    new Paragraph({ spacing: { after: 120 }, children: [] }),
  ];
}
const R = AlignmentType.RIGHT;
const pct = (x, d = 1) => `${(100 * x).toFixed(d)}%`;
const n0 = (x) => Math.round(x).toLocaleString("en-US");
const usd = (x) => "$" + Math.round(x).toLocaleString("en-US");
const usdM = (x) => "$" + (x / 1e6).toFixed(1) + " M";

// ------------------------------------------------------------------ body
const body = [];
const push = (...xs) => xs.forEach((x) => Array.isArray(x) ? body.push(...x) : body.push(x));

// ---- 1. Introduction
push(BREAK(), H1("1. Introduction"),
  P(`Road traffic collisions impose large human and financial costs on cities and on the insurers who cover drivers. For an insurer, two questions matter: how often collisions happen, and how serious they are when they do. A small share of collisions, those involving severe injuries or deaths, accounts for a disproportionate share of claim costs. In the dataset used in this project, the median claim for a fatal collision is about ${D.claim_ratio_fatal_pdo} times that of a property-damage-only collision.`),
  P("Collision risk is also strongly spatial. Crashes cluster at particular intersections and along particular roads, and the conditions that make a crash severe, such as darkness, wet roads and vulnerable road users, vary with time and place. Traditional insurance rating often works with coarse geographic units such as postcodes, which can hide these local patterns."),
  P("This project combines geographic information science and machine learning to address both questions. Collisions are placed precisely on the road network, enriched with information about their surroundings, grouped into hotspots, and used to train models that estimate the probability that a collision will be severe. The results are expressed as a four-tier risk model and as risk zones along individual road segments, and the final model is exposed through a web API so that other systems can use it."),
  H2("1.1 Objectives"),
  BUL("Build a clean, validated collision dataset and place every collision on the road network in a metric coordinate system."),
  BUL("Describe how collision frequency and severity vary by time, driver, vehicle and conditions."),
  BUL("Engineer temporal, environmental, traffic and road-geometry features and remove redundant ones."),
  BUL("Identify dangerous intersections and road corridors with density-based clustering (DBSCAN) instead of K-Means."),
  BUL("Train and compare Logistic Regression, Random Forest and XGBoost for predicting severe or fatal collisions, using methods suited to a rare target."),
  BUL("Choose a decision threshold that improves the detection of severe collisions without flagging every case."),
  BUL("Convert predictions into a four-tier risk model and road-segment risk zones for insurance use."),
  BUL("Deploy the model as an API that can run in a container."),
  H2("1.2 Research Questions"),
  NUM("What share of collisions are severe or fatal, and how much do they cost compared with minor collisions?", "rq"),
  NUM("Do conditions such as night-time, darkness, rain and vehicle type change the likelihood of a severe outcome?", "rq"),
  NUM("Can density-based clustering separate dangerous intersections from dangerous corridors?", "rq"),
  NUM("How well can machine-learning models predict severe collisions using only information available before a crash?", "rq"),
  NUM("Can model outputs be turned into risk tiers and spatial zones that are useful for insurance and road-safety planning?", "rq"),
);

// ---- 2. Literature review
push(H1("2. Literature Review"),
  P("Road-safety research consistently finds that collisions are not randomly distributed in space but concentrate at specific sites and along specific road sections. Hotspot methods range from kernel density estimation to clustering. Because collisions occur on a road network rather than on an open plane, network-based approaches that measure distance along streets are often preferred (Okabe & Sugihara, 2012). Tools such as OSMnx make it practical to download, model and analyse street networks from OpenStreetMap (Boeing, 2017)."),
  P("Among clustering methods, DBSCAN (Ester et al., 1996) suits collision data well. It does not need the number of clusters in advance, it can find clusters of any shape, such as a line of crashes along a road, and it labels isolated events as noise. Its results depend on two parameters, the neighbourhood radius ({i}eps{/i}) and the minimum number of neighbours ({i}min_samples{/i}), which are commonly guided by a k-distance plot. To judge hotspot quality, the Prediction Accuracy Index (PAI) compares the share of events captured with the share of area covered (Chainey et al., 2008)."),
  P("For predicting severity, ensemble methods such as Random Forest (Breiman, 2001) and gradient boosting, including XGBoost (Chen & Guestrin, 2016), are widely used because they capture non-linear effects and interactions, while logistic regression remains a strong and interpretable baseline. Severe collisions are rare, so the problem is imbalanced. Common remedies are oversampling the minority class with SMOTE (Chawla et al., 2002) and cost-sensitive learning through class weights (Lemaître et al., 2017). With imbalanced data, accuracy is misleading, and precision–recall analysis is more informative than ROC analysis alone (Saito & Rehmsmeier, 2015)."),
  P("Spatial data also raises a validation problem: nearby observations resemble each other, so random train–test splits can overstate performance. Spatial or block cross-validation, which keeps neighbouring observations on the same side of a split, gives more realistic estimates (Roberts et al., 2017). Encoding high-cardinality categories such as hotspot identifiers with target statistics must also be done carefully to avoid leaking the label into the features (Micci-Barreca, 2001)."),
  P("Two gaps motivate this project. Many applied studies use centroid clustering such as K-Means, which forces every crash into a round cluster and cannot represent corridors or noise; and many report accuracy on random splits, which hides poor detection of the rare severe class. This project addresses both."),
);

// ---- 3. Study area
push(H1("3. Study Area"),
  P("The study covers the City of Los Angeles, California, within approximately 33.70°–34.34° N and 118.67°–118.16° W. Los Angeles is one of the most car-dependent large cities in the United States, with a dense grid of arterial roads, an extensive freeway system and a wide mix of land uses, from the downtown core and Hollywood nightlife districts to the suburban San Fernando Valley and the port area of San Pedro. The city is divided into 21 Los Angeles Police Department (LAPD) geographic areas, which are used here as reporting units."),
  P(`The collision dataset contains ${n0(D.raw_rows)} records from January 2021 to December 2025. Because no real claims data were available, the records were generated synthetically to resemble LAPD open-data collision records. Crashes concentrate around busy areas, intersections and arterial corridors; follow realistic daily, weekly and seasonal patterns; and carry driver, vehicle, environmental and claim attributes. Severity was generated from known causes such as motorcycles, alcohol, darkness and wet roads, so the true drivers of risk are known and model findings can be checked against them.`),
  P(`All spatial analysis was carried out in WGS 84 / UTM Zone 11N (EPSG:32611), a projected system in metres that covers Los Angeles. The pipeline is designed to use the OpenStreetMap drive network for Los Angeles, downloaded with OSMnx. In the development environment used for this report the OpenStreetMap service could not be reached, so the results were produced on a **synthetic citywide street grid** covering the same extent (${n0(D.grid_nodes)} junctions and ${n0(D.grid_edges)} directed road segments at roughly 300 m spacing). Road names such as “Row 160” in the results therefore refer to grid lines rather than real streets. Re-running the pipeline with network access replaces the grid with the real street network.`),
);

// ---- 4. Methodology
push(H1("4. Methodology"),
  P("The work was organised as a pipeline of five stages, each implemented as a Python module: data generation and ingestion, feature engineering, hotspot clustering, model training, and deployment. The same feature code is used for training and for the deployed API so that both compute features identically."),
  H2("4.1 Dataset Overview"),
  TAB("Dataset overview", ["Attribute", "Value"], [
    ["Records (raw)", n0(D.raw_rows)],
    ["Records after validation", n0(D.clean_rows)],
    ["Period", "January 2021 – December 2025"],
    ["Raw attributes", String(D.raw_cols)],
    ["Spatial fields", "Latitude, Longitude (WGS 84, EPSG:4326)"],
    ["Analysis coordinate system", "UTM Zone 11N, EPSG:32611 (metres)"],
    ["Reporting units", "21 LAPD geographic areas"],
    ["Target field", "High_Risk = 1 for severe-injury or fatal collisions"],
    ["Target prevalence", pct(D.high_risk_rate)],
  ], [3, 5]),
  H2("4.2 Data Cleaning, Reprojection and Road Snapping"),
  P("Each record was checked against a schema and coerced to the correct type. Rows with missing or (0, 0) coordinates, coordinates outside the study area or duplicate identifiers were removed and written to an audit file with the reason, rather than being dropped silently (Table 2). The remaining points were converted to a GeoDataFrame and reprojected from EPSG:4326 to EPSG:32611, so that every distance in later stages is measured in metres."),
  ...TAB("Data validation results", ["Check", "Records"], [
    ["Missing coordinates", n0(D.rejected.null_coordinates)],
    ["Zero (0, 0) coordinates", n0(D.rejected.zero_coordinates)],
    ["Duplicate collision identifier", n0(D.rejected.duplicate_id)],
    ["Outside the study area", n0(D.rejected.outside_study_area)],
    ["**Total rejected**", `**${n0(D.rejected_total)} (${pct(D.rejected_total / D.raw_rows)})**`],
    ["**Retained**", `**${n0(D.clean_rows)}**`],
  ], [5, 3], [null, R]),
  P(`GPS positions are rarely exactly on the road, so each collision was then **snapped** to the road network. The OSMnx graph was projected to the same CRS and its edges, with their true curved geometries, were indexed with a spatial tree. For each collision the nearest road segment was found, and the point was moved to the closest location on that segment using linear referencing. The snap distance was recorded; points more than 50 m from a road were flagged as low confidence and points more than 250 m away were rejected. Each collision also inherited the road attributes of its segment (road class, speed limit, lanes, one-way status and name). On the test grid the median snap distance was ${D.snap_median.toFixed(1)} m and the 95th percentile ${D.snap_p95.toFixed(1)} m; these are larger than expected on real streets because the grid lines are about 300 m apart.`),
  H2("4.3 Feature Engineering"),
  P("New variables were derived from the time, location and surroundings of each collision (Table 3). Hour, weekday and month were also encoded as sine and cosine pairs, so that 23:00 and 00:00 are treated as neighbours rather than 23 hours apart; a second, 12-hour pair lets models represent the two daily rush-hour peaks."),
  ...TAB("Feature engineering summary", ["Group", "Features", "Source"], [
    ["Temporal", "Collision_Hour, DayOfWeek, Month, IsWeekend, IsRushHour, IsNight, WeekendNight; sine/cosine of hour (24 h and 12 h), weekday and month", "Timestamp"],
    ["Weather", "Precipitation (mm/h), visibility (km), days since rain, dry spell before rain, surface friction index", "Simulated historical weather API"],
    ["Traffic", "Congestion index, free-flow and observed speed, speed differential, speed variability", "Simulated traffic API, per road segment and hour"],
    ["Points of interest", "Alcohol-serving venues within 200 m, distance to nearest venue", "Simulated venues (OpenStreetMap option)"],
    ["Road geometry", "Distance to nearest intersection, number of legs, mix of road classes, intersections within 200 m, road class, lanes, segment length", "Road network graph"],
    ["Driver and vehicle", "Driver age, prior claims (3 years), vehicle type, vehicle age", "Collision record"],
    ["Hotspot", "Inside a hotspot, hotspot type, hotspot severe-crash rate", "DBSCAN (Section 4.5)"],
  ], [1.6, 4.6, 2.4]),
  P("The weather, traffic and venue layers stand in for external data services. Each is implemented behind a common interface so that a real service can replace the simulator without changing the rest of the pipeline. Their values are deterministic for a given place and hour, so the same collision always receives the same values in training and in the API."),
  P(`Redundant features were removed with a **Pearson correlation matrix**. For every pair with |r| > 0.85 one feature was dropped, keeping the one more strongly correlated with the target; sine/cosine pairs and the calendar fields Collision_Hour, DayOfWeek and Month were protected. The variance inflation factor (VIF) of the remaining features was then checked, and all were below 10 (highest ${D.vif_max.toFixed(1)}).`),
  H2("4.4 Exploratory Analysis"),
  P("Descriptive statistics, distributions of severity, time of day and driver age, severe-crash rates by condition, and the correlation matrix were used to understand the structure of the data before modelling."),
  H2("4.5 Spatial Clustering with DBSCAN"),
  P("K-Means assigns every collision to one of a fixed number of round clusters. Collision hotspots are neither fixed in number nor round: a dangerous intersection is a tight knot, a dangerous road is a long thin line, and most collisions belong to no hotspot at all. DBSCAN was therefore used. It grows a cluster by linking collisions that each have at least {i}min_samples{/i} neighbours within {i}eps{/i} metres, so it can follow a road of any shape and labels isolated collisions as noise."),
  P("A single pair of parameters cannot find both hotspot shapes: a radius small enough to separate intersections breaks corridors into noise, and one large enough to follow a corridor merges neighbouring intersections. A **two-scale** procedure was used:"),
  BUL(`**Stage 1 – intersections.** A grid search over eps = 15–50 m and min_samples = 6–30 (${D.grid_int} combinations) on all collisions.`),
  BUL(`**Stage 2 – corridors.** A second search over eps = 40–250 m and min_samples = 4–16 (${D.grid_cor} combinations) on the collisions left as noise by stage 1. Only clusters that look like corridors count: at least 4 × eps long, and either straight and narrow (PCA linearity ≥ 0.90, width ≤ 60 m) or following one named road over at least three segments, and at least three times denser than a 500 m ring around them.`),
  P("Each parameter pair was scored by **coverage × ln(PAI)**, where coverage is the share of collisions inside hotspots and PAI is that share divided by the share of city area the hotspots cover (measured as occupied 25 m raster cells). Maximising PAI alone favours ever-smaller knots, while maximising coverage alone joins the whole city into one cluster; the product balances the two. Parameter pairs also had to give a PAI of at least 50, no cluster with more than 5% of all collisions, and at least 10 collisions per hotspot. A k-distance plot was produced as a reference. Each final cluster was then classified as an **intersection**, **corridor** or **area** from its length, shape, dominant road and distance to the nearest junction, and ranked by total insured loss."),
  P("Because the generator planted known intersection and corridor hotspots, the clustering could be checked against ground truth. A planted hotspot counted as recovered when at least half of its collisions fell in clusters made up mainly of that hotspot."),
  H2("4.6 Machine-Learning Models"),
  P(`The target **High_Risk** equals 1 for severe-injury or fatal collisions (${pct(D.high_risk_rate)} of records) and 0 otherwise. Two feature sets were defined. The **underwriting** set uses only information known before a crash: place, time, conditions, driver and vehicle. This is the set a quotation system would see, and it is the model served by the API. The **claims-triage** set adds facts known only after a crash is reported (collision type, number of vehicles, speeding, alcohol, distraction and hit-and-run). Driver gender was kept in the data only for fairness auditing and excluded from all models, as California removed gender as a permitted auto-insurance rating factor in 2019.`),
  P(`Collisions were grouped into 2 km × 2 km blocks and **whole blocks** were assigned to training (80%) or testing (20%), giving ${n0(D.train_rows)} training and ${n0(D.test_rows)} test collisions. Neighbouring collisions share road, lighting and hotspot context, so a random split would leak information and inflate scores. Within the training set, 5-fold grouped cross-validation on the same blocks produced out-of-fold predictions used for model selection, threshold tuning and tier cut points. DBSCAN was refitted on the training blocks only, and test collisions were assigned to those hotspots using DBSCAN's own rule (within eps of a core point).`),
  P("Each model is a single pipeline: impute missing values, encode the hotspot severe-crash rate with a cross-fitted target encoder, one-hot encode categories, oversample with **SMOTE**, then fit the classifier (Table 4). SMOTE runs inside the pipeline, so it only ever sees training folds; validation and test data are never resampled. SMOTE raised the minority class to half the size of the majority class, and cost-sensitive class weights corrected the remaining imbalance. Oversampling all the way to 1:1 would have made the class weights redundant."),
  ...TAB("Model configuration", ["Model", "Configuration"], [
    ["Logistic Regression", "Standardised inputs, C = 0.5; SMOTE only (baseline, no class weights)"],
    ["Random Forest", "400 trees, maximum depth 14, at least 10 samples per leaf; class_weight = 'balanced'"],
    ["XGBoost", "500 trees, learning rate 0.04, maximum depth 5, subsample 0.8; scale_pos_weight = 2 (balanced after SMOTE)"],
    ["All models", "SMOTE to a 1:2 minority:majority ratio on training folds only; model selected by out-of-fold PR-AUC, so the test set is never used to choose"],
  ], [2, 6]),
  H2("4.7 Decision Threshold"),
  P(`A classifier outputs a probability, and a threshold turns it into a yes/no decision. Lowering the threshold catches more severe collisions but flags more cases overall. The threshold was chosen on out-of-fold predictions by maximising the **F2 score**, which weights recall four times as heavily as precision, subject to flagging at most ${pct(D.max_flag_share, 0)} of collisions. Without this cap, a weak signal makes “flag everything” the F2-optimal answer, which is useless in practice. The default 0.50 and a fixed 0.20 were evaluated alongside.`),
  H2("4.8 Four-Tier Risk Model"),
  P("Predicted probabilities were converted into four tiers (**Low, Moderate, High, Very High**) using the 50th, 80th and 95th percentiles of the out-of-fold probabilities as cut points. Using percentiles keeps the tiers stable even though oversampling shifts the raw probabilities upward."),
  P("A second, spatial tier was computed for every road segment with recorded collisions: **expected severe collisions per year** = collisions per year × the segment's average predicted severity, shrunk towards the city average with ten pseudo-collisions (empirical Bayes) so that a segment with a single unlucky crash is not labelled Very High. Segments were tiered with the same percentiles."),
  H2("4.9 Evaluation Metrics"),
  BUL("**PR-AUC** (area under the precision–recall curve): the primary metric; a model with no skill scores the base rate."),
  BUL("**ROC-AUC**: probability that a random severe collision is ranked above a random non-severe one."),
  BUL("**Precision, Recall and F2** at the chosen threshold, and the share of collisions flagged."),
  BUL("**Confusion matrix** at the chosen threshold."),
  BUL("**Brier score**: mean squared error of the probabilities (lower is better)."),
  BUL("For clustering: coverage, **PAI**, and recovery of planted hotspots."),
  H2("4.10 Deployment"),
  P("The trained model, feature definitions, hotspot matcher, threshold, tier cut points and segment tiers are saved together as one bundle. A FastAPI service loads the bundle and road network once at start-up and exposes /predict (one scenario), /predict/batch (up to 500), /model-info and /health. For each request it snaps the location to the road network, computes features with the same code used in training, matches the location to a hotspot and returns the probability, decision, risk tier and location details. The service is packaged with a Dockerfile, and 58 automated tests cover each stage, including a test that the API reproduces the training-time predictions for the same collisions."),
);

// ---- 5. Results
const S = D.severity;
push(H1("5. Results"),
  H2("5.1 Collision Severity Distribution"),
  ...TAB("Collision severity counts and claim costs", ["Severity", "Count", "Percentage", "Median claim"], S.map((r) =>
    [r.label, n0(r.count), pct(r.share), usd(r.median_claim)]), [3, 1.6, 1.6, 2], [null, R, R, R]),
  ...FIG("severity", "Collision severity distribution"),
  P(`Most collisions cause property damage only (${pct(S[0].share)}) or minor injuries (${pct(S[1].share)}). Severe-injury (${pct(S[2].share)}) and fatal (${pct(S[3].share)}) collisions together make up ${pct(D.high_risk_rate)} of the data, which is the High_Risk class. Although rare, they dominate costs: the median fatal claim (${usd(S[3].median_claim)}) is about ${D.claim_ratio_fatal_pdo} times the median property-damage claim (${usd(S[0].median_claim)}).`),
  H2("5.2 Crashes by Hour"),
  ...FIG("hour", "Crashes by hour of day, weekday and weekend"),
  P("On weekdays, collisions follow commuting traffic, with a morning peak at 07:00–08:00 and a larger afternoon peak at 15:00–17:00, when about 8% of weekday collisions occur in a single hour. Weekends have no morning peak, a broad midday plateau, and a much higher share of collisions between midnight and 03:00, consistent with late-night leisure travel."),
  H2("5.3 Driver Age Distribution"),
  ...FIG("age", "Driver age distribution"),
  P("Drivers range from 16 to 92 years old. Most collisions involve drivers aged 25–60, with a separate concentration of young drivers aged about 18–24."),
  H2("5.4 High-Risk Collision Rate by Condition"),
  ...TAB("Share of collisions that are severe or fatal, by condition", ["Condition", "Group", "High-risk rate"], D.conditions.map((r) =>
    [r.cond, r.group, pct(r.rate)]), [2.2, 3.4, 1.6], [null, null, R]),
  P(`Vehicle type has by far the largest effect: ${pct(D.cond_lookup.motorcycle)} of motorcycle and ${pct(D.cond_lookup.bicycle)} of bicycle collisions are severe or fatal, about three times the rate for cars (${pct(D.cond_lookup.sedan)}). Darkness without street lights (${pct(D.cond_lookup.dark_no_lights)}), night-time, rain and young drivers aged 16–24 (${pct(D.cond_lookup.young)}) also raise the rate. Weekday rush-hour collisions are slightly less likely to be severe (${pct(D.cond_lookup.rush)} against ${pct(D.cond_lookup.not_rush)} at other times), mainly because they happen in daylight.`),
  H2("5.5 Correlation Matrix and Collinearity Screening"),
  ...FIG("corr", "Pearson correlation matrix of candidate features (dropped features in red)", 6.0),
  P(`Most features are only weakly correlated with each other. ${D.corr_pairs} pairs exceeded |r| = 0.85, mainly groups that measure the same thing: free-flow speed and speed limit (r = 1.00), driver age and licence years (r = 0.99), and congestion, observed speed, speed ratio and speed differential (|r| = 0.95–0.99). Seven features were dropped, including licence years, speed ratio and the composite intersection complexity score. No single feature is strongly correlated with High_Risk, which suggests that severity results from several factors together.`),
  H2("5.6 Spatial Hotspot Clusters"),
  ...FIG("kdist", "k-distance curve (k = 16) with the selected eps for each stage", 5.2),
  P(`The k-distance curve shows that most collisions have their 16th-nearest neighbour hundreds of metres away, while a minority, those inside hotspots, have it within a few tens of metres. The automatic “knee” of the curve (about ${n0(D.knee)} m) reflects the spacing of background collisions, which is why the radius was chosen by the PAI-based grid search rather than by the knee alone.`),
  ...TAB("Selected DBSCAN parameters and hotspot recovery", ["Measure", "Intersections (stage 1)", "Corridors (stage 2)"], [
    ["Selected eps", `${D.db.int.eps} m`, `${D.db.cor.eps} m`],
    ["Selected min_samples", String(D.db.int.ms), String(D.db.cor.ms)],
    ["Clusters found", String(D.db.int.n), String(D.db.cor.n)],
    ["Share of collisions captured", pct(D.db.int.coverage), pct(D.db.cor.coverage)],
    ["PAI", n0(D.db.int.pai), n0(D.db.cor.pai)],
    ["Planted hotspots detectable", String(D.db.eval.intersection_detectable), String(D.db.eval.corridor_detectable)],
    ["Planted hotspots recovered", `${D.db.int_recovered} (${pct(D.db.eval.intersection_recall)})`, `${D.db.cor_recovered} (${pct(D.db.eval.corridor_recall)})`],
  ], [3, 2.5, 2.5], [null, R, R]),
  ...FIG("hotspot", "DBSCAN crash hotspots by type (grey = noise)", 5.0),
  P(`The two stages produced ${D.db.total} hotspots: **${D.db.types.intersection} intersections, ${D.db.types.corridor} corridors and ${D.db.types.area} areas**. Together they contain ${pct(D.db.coverage_total)} of collisions on about ${D.db.area_km2.toFixed(1)} km² of road footprint and account for ${pct(D.db.claims_share, 0)} of all insured losses. Every detectable planted intersection and ${D.db.cor_recovered} of ${D.db.eval.corridor_detectable} planted corridors were recovered; ${pct(D.db.eval.cluster_precision)} of clusters correspond to a real planted hotspot, ${pct(D.db.eval.type_accuracy)} were given the correct type, and only ${pct(D.db.eval.background_in_hotspots)} of background collisions were wrongly placed in a hotspot. The map shows intersections as compact knots and corridors as straight runs along arterials, with most background collisions correctly left as noise.`),
  ...TAB("Top hotspots by total insured loss", ["Rank", "Type", "Collisions", "Length", "Road", "Severe share", "Total claims"], D.top_hotspots.map((h) =>
    [String(h.loss_rank), h.cluster_type, n0(h.n_collisions), `${n0(h.length_m)} m`, h.dominant_road, pct(h.severe_fatal_share), usdM(h.total_claims_usd)]),
    [0.8, 1.6, 1.3, 1.2, 1.3, 1.4, 1.6], [R, null, R, R, null, R, R], 18),
  P(`Corridors dominate the top of the loss ranking because they are long and collect many collisions. However, the share of severe or fatal collisions inside hotspots (${pct(D.db.severe_share_in_hotspots)}) is almost the same as outside them (${pct(D.db.severe_share_outside)}). Hotspots concentrate **how often** collisions happen, not **how serious** they are.`),
  H2("5.7 Model Comparison"),
  ...TAB("Model comparison on held-out blocks (underwriting feature set)", ["Model", "OOF PR-AUC", "Test PR-AUC", "Test ROC-AUC", "Brier", "Threshold", "Recall", "Precision"], D.models.map((m) =>
    [m.name, m.oof_pr_auc.toFixed(3), m.test_pr_auc.toFixed(3), m.test_roc_auc.toFixed(3), m.test_brier.toFixed(3), m.threshold.toFixed(2), pct(m.recall), pct(m.precision)]),
    [1.9, 1.1, 1.1, 1.1, 0.85, 1.2, 0.9, 1.2], [null, R, R, R, R, R, R, R], 18),
  ...FIG("models", "Model comparison on held-out blocks"),
  ...FIG("pr", "Precision–recall curves on held-out blocks (dots mark the chosen thresholds)", 5.2),
  P(`All three models beat the no-skill baseline (PR-AUC ${D.base_test.toFixed(3)}), but only modestly. Random Forest scored highest on the test blocks (PR-AUC ${D.models[1].test_pr_auc.toFixed(3)}, ROC-AUC ${D.models[1].test_roc_auc.toFixed(3)}). Logistic Regression was selected because it scored highest on out-of-fold PR-AUC (${D.models[0].oof_pr_auc.toFixed(3)}), and the test set was not used for selection. The gaps between models are small. The precision–recall curves show useful precision (25–40%) only for the few highest-scoring collisions, and fall towards the base rate as recall increases. XGBoost had the best Brier score; oversampling inflates the raw probabilities of the other models, which is why the risk tiers use percentiles.`),
  P(`Three ways of handling the imbalance were also compared. For Logistic Regression, test PR-AUC was ${D.os.smote.lr.toFixed(3)} with SMOTE, ${D.os.smotenc.lr.toFixed(3)} with SMOTENC and ${D.os.none.lr.toFixed(3)} with class weights only; for Random Forest it was ${D.os.smote.rf.toFixed(3)}, ${D.os.smotenc.rf.toFixed(3)} and ${D.os.none.rf.toFixed(3)}. SMOTENC, the SMOTE variant for mixed data, performed worst: it gives each synthetic record the most common category among its neighbours, which turned synthetic severe crashes into mostly cars and diluted the motorcycle signal. Class weights alone performed about as well as SMOTE, so on this data oversampling added little beyond reweighting. SMOTE on the one-hot-encoded data was kept, as specified for the project, and combined with class weights.`),
  H2("5.8 Decision Threshold"),
  ...TAB("Effect of the decision threshold (Logistic Regression, test blocks)", ["Threshold", "Recall", "Precision", "F2", "Collisions flagged"], D.thresholds.map((t) =>
    [t.label, pct(t.recall), pct(t.precision), t.f2.toFixed(3), pct(t.flagged)]), [2.6, 1.3, 1.3, 1.1, 1.9], [null, R, R, R, R]),
  ...FIG("confusion", `Confusion matrix, Logistic Regression at threshold ${D.thr.toFixed(2)}`, 4.2),
  P(`The default threshold of 0.50 caught only ${pct(D.thresholds[0].recall)} of severe collisions. A fixed threshold of 0.20 caught ${pct(D.thresholds[1].recall)}, but only by flagging ${pct(D.thresholds[1].flagged)} of all collisions, at which point precision (${pct(D.thresholds[1].precision)}) is barely above the base rate. The tuned threshold of ${D.thr.toFixed(2)} caught ${pct(D.thresholds[2].recall)} of severe collisions (${D.conf.tp} of ${D.conf.tp + D.conf.fn}) while flagging ${pct(D.thresholds[2].flagged)}. With an oversampled model, a fixed threshold such as 0.20 should not be applied without checking how many cases it flags.`),
  H2("5.9 Underwriting versus Claims-Triage Features"),
  ...TAB("Best test performance by feature set, and the best achievable", ["Feature set", "Best PR-AUC", "Best ROC-AUC", "Ceiling PR-AUC", "Ceiling ROC-AUC"], D.feature_sets.map((f) =>
    [f.name, f.pr.toFixed(3), f.roc.toFixed(3), f.ceil_pr.toFixed(3), f.ceil_roc.toFixed(3)]), [2.4, 1.4, 1.4, 1.6, 1.6], [null, R, R, R, R]),
  P(`Adding post-crash facts raised PR-AUC from ${D.feature_sets[0].pr.toFixed(3)} to ${D.feature_sets[1].pr.toFixed(3)} and ROC-AUC from ${D.feature_sets[0].roc.toFixed(3)} to ${D.feature_sets[1].roc.toFixed(3)}. The “ceiling” columns show the best any model could achieve: they score each collision with the generator's own severity formula, which is the true cause of severity, before random noise. The models come within about 0.02 of this ceiling for both feature sets. The modest scores therefore reflect how much randomness the data contain, not a weakness of the methods.`),
  H2("5.10 Feature Importance"),
  ...FIG("importance", "Permutation feature importance, Logistic Regression (test blocks)"),
  ...TAB("Feature importance ranking (drop in test PR-AUC when shuffled)", ["Rank", "Feature", "Importance", "Std. dev."], D.importance.map((r, i) =>
    [String(i + 1), r.label, r.mean.toFixed(4), r.std.toFixed(4)]), [0.8, 4, 1.6, 1.6], [R, null, R, R]),
  P(`Vehicle type is by far the most important feature, followed by lighting, night-time, surface friction (wet roads) and vehicle age. Driver age has a small and uncertain effect. This ranking matches the causes used to generate severity, which confirms that the models learned real relationships rather than artefacts. Location features contribute little to severity: hotspot membership ranks only ${D.importance_hotspot_rank.in_hotspot}th, consistent with the finding in Section 5.6.`),
  H2("5.11 Four-Tier Risk Model"),
  ...TAB("Four-tier risk model on held-out blocks", ["Tier", "Collisions", "Share of collisions", "High-risk rate", "Share of severe", "Lift"], D.tiers.map((t) =>
    [t.tier, n0(t.crashes), pct(t.share_of_crashes), pct(t.high_risk_rate), pct(t.share_of_severe), `${t.lift.toFixed(2)}×`]),
    [1.5, 1.3, 1.7, 1.5, 1.5, 1], [null, R, R, R, R, R], 18),
  ...FIG("tiers", "Observed severe/fatal rate by risk tier on held-out blocks", 5.2),
  P(`The tiers are correctly ordered on data the model never saw. Collisions in the **Very High** tier were severe or fatal ${pct(D.tiers[3].high_risk_rate)} of the time, ${D.tiers[3].lift.toFixed(1)} times the base rate, compared with ${pct(D.tiers[0].high_risk_rate)} in the **Low** tier. The Very High tier contains ${pct(D.tiers[3].share_of_crashes)} of collisions but ${pct(D.tiers[3].share_of_severe)} of severe ones.`),
  H2("5.12 Insurance Risk Zones"),
  ...FIG("segments", "Road-segment risk tiers (expected severe collisions per year)", 5.0),
  P(`Of the ${n0(D.segments.total)} road segments with recorded collisions, ${n0(D.segments.Low)} were classified Low, ${n0(D.segments.Moderate)} Moderate, ${n0(D.segments.High)} High and ${n0(D.segments["Very High"])} Very High. Very High segments coincide with the DBSCAN intersections (shown as crosses) and corridors (straight lines), and with the densest areas of background collisions. Because segment risk combines frequency with severity, it is driven mainly by how many collisions a segment has.`),
  ...TAB("Top LAPD areas by total insured loss", ["Area", "Collisions", "High-risk rate", "Total claims", "Mean predicted risk*"], D.areas.map((a) =>
    [a.area_name, n0(a.crashes), pct(a.high_risk_rate), usdM(a.total_claims), a.mean_predicted_risk_test == null ? "–" : a.mean_predicted_risk_test.toFixed(3)]),
    [2, 1.4, 1.5, 1.6, 2], [null, R, R, R, R]),
  P("{i}*Average model probability for test-block collisions in the area; areas without test collisions show a dash. Probabilities are raised by oversampling and are comparable only with each other.{/i}", { spacing: { after: 160 } }),
  P(`Area totals are driven by collision counts: Pacific and Harbor rank highest because they collect many collisions, while severe-crash rates vary only between ${pct(D.area_rate_min)} and ${pct(D.area_rate_max)} across the top ten areas. For pricing, the location of a policy therefore mostly informs claim **frequency**, while vehicle, driver and conditions inform **severity**.`),
  H2("5.13 Deployment"),
  P(`The final model is served by a FastAPI application. A request with a location, time, driver age, vehicle and weather returns the probability of a severe outcome, the high-risk decision, the risk tier, the snapped road, the matched hotspot and the segment's spatial tier. Requests containing driver gender or unknown fields are rejected, as are locations outside Los Angeles or more than 250 m from any road. After the road index and other spatial indexes were built once at start-up rather than on every request, a single prediction took about ${D.latency_ms} ms on the development machine, down from about two seconds. All 58 automated tests pass. The Docker image could not be built in the development environment because the container registry was blocked, so the container start-up was reproduced with a clean installation from the requirements file and the Dockerfile's start command, and its health check passed.`),
);

// ---- 6. Discussion
push(H1("6. Discussion"),
  P(`The descriptive analysis shows that the dataset is dominated by property-damage and minor-injury collisions, with severe or fatal outcomes in only ${pct(D.high_risk_rate)} of cases. Accuracy would therefore be misleading: a model that never predicts a severe collision would be right 88.5% of the time. This is why PR-AUC, recall and confusion matrices were used, and why the study handled the imbalance with SMOTE inside the training folds, class weights and a tuned threshold.`),
  P("The spatial analysis shows that collisions concentrate strongly in a small part of the road network: about 37% of collisions and 38% of insured losses fall in 128 hotspots covering roughly 1.2 km². The two-scale DBSCAN separated compact intersection hotspots from long corridor hotspots, which a single parameter setting or K-Means could not do, and left most background collisions as noise. However, hotspots did not have a higher share of severe outcomes than the rest of the city. Location tells an insurer where collisions are frequent, but not where they are more serious."),
  P("Severity was driven mainly by who and what was involved and under what conditions: motorcycles and bicycles, darkness, wet roads, older vehicles and young drivers. Using only information available before a crash, the models reached PR-AUC 0.16–0.18 against a baseline of 0.11. This is within about 0.02 of the best any model could achieve on this data. Adding post-crash details raised performance, which supports using a separate claims-triage model once a claim is reported."),
  P("The choice of decision threshold mattered more than the choice of model. The three models differed by about 0.01 in PR-AUC, whereas moving the threshold from 0.50 to 0.20 changed the share of collisions flagged from 6% to 93%. Tuning the threshold to recall-weighted F2 with a cap on the flagged share gave a usable balance. The four-tier risk model turned the probabilities into ordered groups that separate risk clearly at the top end, where an insurer's attention is most valuable."),
  P("Several problems were found and corrected during the work, and they are useful lessons. A leave-one-out target encoding of hotspot identifiers leaked the label and made one model predict high risk for almost everything; it was replaced with cross-fitted encoding. SMOTENC weakened the categorical signal it was meant to preserve. A simulated weather feature initially depended on a row's position in a batch, which would have made API predictions differ from training; it was keyed to place and time instead."),
  H2("6.1 Limitations"),
  BUL("The collision records are synthetic. Patterns were designed to be realistic, but results must be confirmed on real LAPD and claims data."),
  BUL("Results were produced on a synthetic street grid because OpenStreetMap could not be reached; snapping distances, road names and segment tiers will differ on the real network."),
  BUL("Weather, traffic and venue features were simulated; real data services may show stronger or weaker effects."),
  BUL("Severity in the data contains a large random component, which limits how well any model can perform."),
  BUL("Oversampling distorts predicted probabilities; tiers rely on percentiles, but probabilities should be calibrated before being used as prices."),
  BUL("There is no exposure measure such as vehicle-kilometres travelled, so segment frequency reflects traffic volume as well as danger."),
  BUL("Results come from a single spatial train–test split; repeated spatial cross-validation would give confidence intervals."),
  H2("6.2 Practical Implications"),
  BUL("**Location-based frequency loading:** road-segment and hotspot tiers indicate where claims are likely to be frequent."),
  BUL("**Severity rating:** vehicle type, vehicle age, driver age and typical driving conditions indicate how costly a claim is likely to be."),
  BUL("**Claims triage:** the post-crash model can prioritise reported claims that are likely to involve serious injury."),
  BUL("**Road-safety planning:** intersection hotspots point to junction redesign; corridor hotspots point to speed management and lighting."),
  BUL("**Portfolio monitoring:** policies can be tracked by the risk tier of the roads where their vehicles are driven."),
  BUL("**Real-time use:** the API returns a risk assessment in about a tenth of a second, which is suitable for quotation systems."),
);

// ---- 7. Conclusion
push(H1("7. Conclusion"),
  P("This project developed an end-to-end geospatial system for assessing vehicle collision insurance risk in Los Angeles. Collisions were validated, reprojected to a metric coordinate system and snapped to the road network; enriched with temporal, weather, traffic, venue and road-geometry features; screened for redundancy; grouped into hotspots with a two-scale DBSCAN; and used to train and compare three imbalance-aware classification models under spatial validation. The results were turned into a four-tier risk model and road-segment risk zones and deployed as a tested web API with a container definition."),
  P(`The analysis showed that severe or fatal collisions are rare (${pct(D.high_risk_rate)}) but costly; that collisions concentrate in a small number of intersection and corridor hotspots, which capture ${pct(D.db.coverage_total)} of collisions and ${pct(D.db.claims_share, 0)} of losses; that severity depends mainly on vehicle type, lighting and road conditions rather than location; and that a tuned threshold and percentile-based tiers turn a modest model into a usable risk ranking, with the Very High tier carrying ${D.tiers[3].lift.toFixed(1)} times the base rate of severe outcomes.`),
  P("Future work should apply the pipeline to real LAPD collision records and the real OpenStreetMap network; replace the simulated enrichment layers with real weather and traffic services; add exposure data; calibrate predicted probabilities; repeat the evaluation with multiple spatial folds; compare DBSCAN with HDBSCAN; and model claim frequency and claim cost directly (for example with Poisson or Tweedie models) for pricing."),
);

// ---- 8. References
push(H1("8. References"));
D.references.forEach((r) => push(new Paragraph({ children: runs(r), spacing: { after: 120 },
  indent: { left: 480, hanging: 480 } })));

// ---- 9. Annexure
push(H1("9. Annexure"),
  H2("Annexure A: Python Workflow Summary"),
  BUL("Generate (or load) collision records: python -m src.data_generator"),
  BUL("Validate, reproject to EPSG:32611 and snap to the road network: python -m src.ingestion"),
  BUL("Engineer features and run collinearity screening: python -m src.features"),
  BUL("Detect hotspots with two-scale DBSCAN: python -m src.clustering"),
  BUL("Train and evaluate models, tune the threshold, build risk tiers: python -m src.train"),
  BUL("Serve the model: uvicorn api.app:app, or build and run the Docker image"),
  H2("Annexure B: Exported Outputs"),
  ...TAB("Main output files", ["File", "Contents"], [
    ["data/processed/la_collisions_snapped.parquet", "Cleaned, projected, road-snapped collisions"],
    ["data/interim/la_collisions_rejected.csv", "Rejected records with reasons"],
    ["data/processed/la_collisions_features.parquet", "All engineered features"],
    ["data/processed/la_hotspot_clusters.parquet", "One polygon per hotspot with type, road and losses"],
    ["data/processed/la_segment_risk_tiers.parquet", "Road-segment risk tiers"],
    ["models/risk_model.joblib", "Underwriting model bundle used by the API"],
    ["models/risk_model_claims_triage.joblib", "Claims-triage model bundle"],
    ["reports/model_comparison.csv, risk_tiers_test.csv, threshold_metrics.json", "Evaluation results"],
    ["reports/dbscan_tuning_grid.csv, dbscan_groundtruth_eval.csv", "Clustering search and recovery"],
  ], [4.6, 3.4]),
  H2("Annexure C: Example API Request and Response"),
  P("Request to POST /predict:"),
  ...CODE(D.api_request),
  P("Response:", { spacing: { before: 160, after: 80 } }),
  ...CODE(D.api_response),
);

// ---- 10. Appendix
push(H1("10. Appendix"),
  H2("Appendix A: Key Libraries"),
  ...TAB("Python libraries used", ["Library", "Version", "Purpose"], D.libraries, [2, 1.2, 4.8]),
  H2("Appendix B: Recommended Future Variables"),
  BUL("Real LAPD collision records and insurer claim histories"),
  BUL("Measured weather at the time of each collision"),
  BUL("Traffic volume and speed from road sensors or probe data"),
  BUL("Exposure: vehicle-kilometres travelled per segment and per policy"),
  BUL("Driver history: violations, years licensed, telematics"),
  BUL("Road design: speed cameras, signal timing, lighting inventory, pedestrian crossings"),
  BUL("Claim payment and settlement time"),
);

// ------------------------------------------------------------------ front matter
// Built after the body so figure/table lists match the numbering exactly.
const front = [];
front.push(
  new Paragraph({ spacing: { before: 2400, after: 240 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "SPATIAL RISK ASSESSMENT FOR VEHICLE COLLISION INSURANCE IN LOS ANGELES", bold: true, size: 40 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 600 },
    border: { bottom: { style: BorderStyle.SINGLE, size: 8, color: "4F81BD", space: 8 } },
    children: [new TextRun({ text: "Road-network snapping, two-scale DBSCAN hotspot detection, imbalance-aware severity modelling and a deployable risk API", italics: true, size: 26 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 120 }, children: [new TextRun({ text: "Project Report", size: 28 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "October 2026", size: 24 })] }),
  BREAK(),
  new Paragraph({ spacing: { after: 240 }, children: [new TextRun({ text: "INDEX", bold: true, size: 28 })] }),
  ...["I. List of Figures", "II. List of Tables", "III. Abbreviation List", "IV. Acknowledgement", "V. Preface", "VI. Abstract",
    "1. Introduction", "2. Literature Review", "3. Study Area", "4. Methodology", "5. Results", "6. Discussion",
    "7. Conclusion", "8. References", "9. Annexure", "10. Appendix"].map((t) => PLAIN(t, { spacing: { after: 100 } })),
  BREAK(),
  H1("I. List of Figures"),
  ...FIGS.map((c, i) => PLAIN(`Figure ${i + 1}. ${c}`)),
  H1("II. List of Tables"),
  ...TABS.map((c, i) => PLAIN(`Table ${i + 1}. ${c}`)),
  BREAK(),
  H1("III. Abbreviation List"),
  ...D.abbreviations.map(([a, b]) => PLAIN(`${a} – ${b}`)),
  BREAK(),
  H1("IV. Acknowledgement"),
  P("I would like to thank my project guide and the faculty of my university for their guidance and feedback throughout this project. Their suggestions shaped how the problem was framed, how the models were evaluated and how the results are presented in this report."),
  P("I am grateful to my university for providing the facilities and learning environment needed to complete this work, and to the developers of the open-source tools used throughout, including Python, GeoPandas, OSMnx, scikit-learn, imbalanced-learn, XGBoost and FastAPI, without which a project of this scope would not have been possible."),
  P("Finally, I thank my family and friends for their encouragement and support while I worked on this project."),
  H1("V. Preface"),
  P(`This report, titled “Spatial Risk Assessment for Vehicle Collision Insurance in Los Angeles”, documents a project that builds an end-to-end system for estimating how likely a road collision is to be severe or fatal, and where in the city that risk is concentrated. The work uses a ${n0(D.raw_rows)}-record collision dataset for the City of Los Angeles.`),
  P("The project was carried out in five stages: building and cleaning the dataset and placing each collision on the road network; engineering temporal, environmental and road-geometry features; detecting crash hotspots with density-based clustering; training and comparing machine-learning models for severe-collision prediction; and packaging the final model as a web service that can run in a container."),
  P("The report follows the same order. It introduces the problem, reviews relevant literature, describes the study area and data, explains the methods, presents the results and discusses what they mean for insurers and road-safety planners, including the limits of what the data can support. All work was carried out in Python."),
  P("Through this project I gained practical experience in geospatial data preparation, coordinate reference systems, spatial clustering, imbalanced classification, careful model evaluation and deploying a model as an API."),
  BREAK(),
  H1("VI. Abstract"),
  P(D.abstract),
);

// ------------------------------------------------------------------ document
const doc = new Document({
  creator: "LA Collision Risk Project",
  title: "Spatial Risk Assessment for Vehicle Collision Insurance in Los Angeles",
  styles: {
    default: { document: { run: { font: FONT, size: 24 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: HFONT, size: 28, bold: true, color: "365F91" },
        paragraph: { spacing: { before: 480, after: 160 }, keepNext: true, keepLines: true, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: HFONT, size: 26, bold: true, color: "4F81BD" },
        paragraph: { spacing: { before: 280, after: 120 }, keepNext: true, keepLines: true, outlineLevel: 1 } },
    ],
  },
  numbering: { config: [
    { reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
      style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
    { reference: "numbers", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
      style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
    { reference: "rq", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
      style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
  ] },
  sections: [{
    properties: { page: { size: { width: 12240, height: 15840 },
      margin: { top: 1440, right: 1800, bottom: 1440, left: 1800, header: 720, footer: 720 } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
      children: [new TextRun({ children: [PageNumber.CURRENT], size: 20 })] })] }) },
    children: [...front, ...body],
  }],
});

Packer.toBuffer(doc).then((buf) => { fs.writeFileSync(OUT, buf); console.log("wrote", OUT, figNo, "figures", tabNo, "tables"); });
