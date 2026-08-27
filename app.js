/*
 * Local browser inference for the frozen MIND SGD logistic model.  The export
 * preserves sklearn's feature order, StandardScaler parameters and OneHotEncoder
 * vocabulary. Unknown categories produce no one-hot feature, matching
 * handle_unknown="ignore" in the accepted preprocessing contract.
 */
(async function initialise() {
  const [model, examples] = await Promise.all([
    fetch("web_model.json").then((response) => response.ok ? response.json() : Promise.reject(new Error("web_model.json could not be loaded"))),
    fetch("examples.json").then((response) => response.ok ? response.json() : Promise.reject(new Error("examples.json could not be loaded"))),
  ]);

  const form = document.querySelector("#prediction-form");
  const simpleControls = document.querySelector("#simple-controls");
  const advancedControls = document.querySelector("#advanced-controls-grid");
  const exampleButtons = document.querySelector("#example-buttons");
  const scoreElement = document.querySelector("#score");
  const bandElement = document.querySelector("#score-band");
  const baselineElement = document.querySelector("#baseline");
  const indexElement = document.querySelector("#relative-index");
  const interpretationElement = document.querySelector("#interpretation");
  const positiveList = document.querySelector("#positive-contributors");
  const negativeList = document.querySelector("#negative-contributors");
  const inputByName = new Map();
  const officialDefaultExample = examples[1];
  let selectedExample = officialDefaultExample;
  const calendarDays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"];

  const labels = {
    candidate_set_size: "Candidate set size", history_length: "History length",
    prior_category_count: "Prior category count", prior_subcategory_count: "Prior subcategory count",
    category_affinity: "Category affinity", subcategory_affinity: "Subcategory affinity",
    most_recent_category_match: "Most recent category match", most_recent_subcategory_match: "Most recent subcategory match",
    hour: "Hour", title_length: "Title length", abstract_length: "Abstract length",
    abstract_missing: "Abstract missing", short_or_cold_history: "Short or cold history",
    candidate_category: "Candidate category", candidate_subcategory: "Candidate subcategory", day_of_week: "Day of week",
  };
  const simpleNames = new Set(["candidate_category", "candidate_subcategory", "candidate_set_size", "history_length", "category_affinity", "subcategory_affinity"]);

  function fieldContainer(name, element, description) {
    const container = document.createElement("div");
    container.className = "field" + (name === "candidate_subcategory" ? " full" : "");
    const label = document.createElement("label");
    label.htmlFor = name;
    label.textContent = labels[name];
    container.append(label, element);
    if (description) { const hint = document.createElement("small"); hint.textContent = description; container.append(hint); }
    return container;
  }

  function createSelect(spec) {
    const select = document.createElement("select");
    select.name = spec.name; select.id = spec.name;
    const choices = spec.name === "day_of_week" ? calendarDays : spec.categories;
    choices.forEach((category) => {
      const option = document.createElement("option"); option.value = category; option.textContent = category; select.append(option);
    });
    inputByName.set(spec.name, select);
    const container = fieldContainer(spec.name, select, spec.name === "day_of_week" ? "Checking training vocabulary…" : "Train-fitted categories only");
    if (spec.name === "day_of_week") container.querySelector("small").id = "day-of-week-training-status";
    return container;
  }

  function createNumeric(spec) {
    if (spec.kind === "binary") {
      const input = document.createElement("input");
      input.type = "checkbox"; input.name = spec.name; input.id = spec.name;
      input.checked = false;
      inputByName.set(spec.name, input);
      const container = fieldContainer(spec.name, input, "");
      container.classList.add("toggle-field");
      container.querySelector("label").textContent = labels[spec.name];
      container.querySelector("label").prepend(input);
      return container;
    }
    const input = document.createElement("input");
    input.type = "number"; input.name = spec.name; input.id = spec.name; input.step = "any";
    input.min = "0";
    input.value = String(spec.mean);
    input.required = true;
    inputByName.set(spec.name, input);
    return fieldContainer(spec.name, input, "Allowed model input");
  }

  model.categorical_features.forEach((spec) => {
    (simpleNames.has(spec.name) ? simpleControls : advancedControls).append(createSelect(spec));
  });
  model.numeric_features.forEach((spec) => {
    (simpleNames.has(spec.name) ? simpleControls : advancedControls).append(createNumeric(spec));
  });

  function activeInputs() {
    const values = {};
    model.numeric_features.forEach((spec) => {
      const element = inputByName.get(spec.name);
      const value = spec.kind === "binary" ? (element.checked ? 1 : 0) : Number(element.value);
      values[spec.name] = Number.isFinite(value) ? value : spec.mean;
    });
    model.categorical_features.forEach((spec) => { values[spec.name] = inputByName.get(spec.name).value; });
    return values;
  }

  function rawModelScore(values) {
    let linear = model.model.intercept;
    const contributions = [];
    model.numeric_features.forEach((spec, index) => {
      const standardized = (values[spec.name] - spec.mean) / spec.scale;
      const contribution = model.model.coefficients[index] * standardized;
      linear += contribution;
      contributions.push({ label: labels[spec.name], contribution });
    });
    model.categorical_features.forEach((spec) => {
      const categoryIndex = spec.categories.indexOf(values[spec.name]);
      if (categoryIndex >= 0) {
        const contribution = model.model.coefficients[spec.offset + categoryIndex];
        linear += contribution;
        contributions.push({ label: `${labels[spec.name]}: ${values[spec.name]}`, contribution });
      }
    });
    return { score: 1 / (1 + Math.exp(-linear)), contributions };
  }

  function bandFor(relativeIndex) {
    if (relativeIndex < 0.75) return "Below Baseline";
    if (relativeIndex < 1.25) return "Near Baseline";
    if (relativeIndex < 2) return "Above Baseline";
    return "Strongly Above Baseline";
  }

  function contributorItem(item, direction) {
    const listItem = document.createElement("li");
    const name = document.createElement("span"); name.textContent = item.label;
    const amount = document.createElement("span"); amount.className = direction;
    amount.textContent = `${item.contribution >= 0 ? "+" : ""}${item.contribution.toFixed(3)}`;
    listItem.append(name, amount);
    return listItem;
  }

  function renderContributors(contributions) {
    const positive = contributions.filter((item) => item.contribution > 0).sort((a, b) => b.contribution - a.contribution).slice(0, 3);
    const negative = contributions.filter((item) => item.contribution < 0).sort((a, b) => a.contribution - b.contribution).slice(0, 3);
    positiveList.replaceChildren(...(positive.length ? positive.map((item) => contributorItem(item, "positive")) : [emptyContributor("No positive active contribution")]));
    negativeList.replaceChildren(...(negative.length ? negative.map((item) => contributorItem(item, "negative")) : [emptyContributor("No negative active contribution")]));
  }

  function emptyContributor(message) { const element = document.createElement("li"); element.textContent = message; return element; }

  function updateDayOfWeekStatus() {
    const daySpec = model.categorical_features.find((spec) => spec.name === "day_of_week");
    const status = document.querySelector("#day-of-week-training-status");
    if (!daySpec || !status) return;
    const selectedDay = inputByName.get("day_of_week").value;
    status.textContent = daySpec.categories.includes(selectedDay)
      ? "Seen in training — active one-hot feature."
      : "Unseen in training — safely ignored by the fitted categorical encoder.";
  }

  function update() {
    updateDayOfWeekStatus();
    const { score, contributions } = rawModelScore(activeInputs());
    const baseline = model.baseline.dev_positive_prevalence;
    const relative = score / baseline;
    const band = bandFor(relative);
    scoreElement.textContent = `${(score * 100).toFixed(2)}%`;
    baselineElement.textContent = `${(baseline * 100).toFixed(2)}%`;
    indexElement.textContent = `${relative.toFixed(2)}×`;
    bandElement.textContent = band;
    interpretationElement.textContent = `Under this candidate and history context, the model assigns an engagement score ${band} at ${relative.toFixed(2)}× the official development-set baseline. It is a model estimate, not an outcome claim.`;
    renderContributors(contributions);
  }

  function loadExample(example) {
    selectedExample = example;
    Object.entries(example.inputs).forEach(([name, value]) => {
      if (inputByName.has(name)) {
        const element = inputByName.get(name);
        if (element.type === "checkbox") element.checked = Number(value) === 1;
        else element.value = String(value);
      }
    });
    [...exampleButtons.querySelectorAll("button")].forEach((button) => button.classList.toggle("is-active", button.dataset.exampleId === example.id));
    update();
  }

  examples.forEach((example) => {
    const button = document.createElement("button");
    button.type = "button"; button.className = "example-button"; button.dataset.exampleId = example.id; button.textContent = example.label;
    button.addEventListener("click", () => loadExample(example));
    exampleButtons.append(button);
  });
  form.addEventListener("submit", (event) => { event.preventDefault(); update(); });
  form.addEventListener("input", update);
  form.addEventListener("change", update);
  document.querySelector("#reset-button").addEventListener("click", () => loadExample(officialDefaultExample));
  loadExample(officialDefaultExample);
})().catch((error) => {
  document.querySelector("#interpretation").textContent = "The local model artifact could not be loaded. Start the included local static preview server and reload this page.";
  console.error("STRONGR OS model demo initialisation failed", error);
});
