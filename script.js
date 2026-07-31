// ─── Logistic Regression Model ───────────────────────────────────
// Replace these coefficients after training on your dataset (Python/sklearn)
const MODEL = {
  intercept: -2.5,
  weights: {
    age: 0.045,
    sex: 0.8,
    cp: 0.55,
    trestbps: 0.012,
    chol: 0.003,
    fbs: 0.4,
    restecg: 0.35,
    thalach: -0.018,
    exang: 0.9,
  },
  threshold: 0.5,
};

function sigmoid(z) {
  return 1 / (1 + Math.exp(-z));
}

function predictHeartDisease(data) {
  let z = MODEL.intercept;
  for (const [key, weight] of Object.entries(MODEL.weights)) {
    z += weight * Number(data[key]);
  }
  const probability = sigmoid(z);
  const isHighRisk = probability >= MODEL.threshold;
  return { probability, isHighRisk };
}

// Tries the real Flask/scikit-learn backend first (POST /api/predict).
// If the backend isn't running (e.g. the page was opened directly as a
// file, or the server is down), falls back to the local JS model above
// so the page still works standalone.
async function getPrediction(data) {
  const fields = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
  ];
  const payload = {};
  fields.forEach((f) => (payload[f] = Number(data[f])));

  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) throw new Error(`Backend responded ${res.status}`);
    const result = await res.json();
    console.info("Prediction served by backend model (scikit-learn).");
    return { probability: result.probability, isHighRisk: result.risk === "high", source: "backend" };
  } catch (err) {
    console.warn("Backend unavailable, using local fallback model:", err.message);
    return { ...predictHeartDisease(data), source: "local" };
  }
}

// ─── Recommendations Engine ──────────────────────────────────────
function getRecommendations(data, isHighRisk) {
  const recs = [];
  const lifestyle = [];

  if (Number(data.age) > 50) {
    recs.push("Schedule annual cardiac check-ups due to age-related risk factors.");
  }
  if (Number(data.trestbps) > 140) {
    recs.push("Your resting blood pressure is elevated. Monitor it daily and reduce salt intake.");
    lifestyle.push("Follow a DASH diet rich in fruits, vegetables, and whole grains.");
  }
  if (Number(data.chol) > 240) {
    recs.push("High cholesterol detected. Consider lipid panel follow-up with your physician.");
    lifestyle.push("Reduce saturated fats; include oats, nuts, and fatty fish in your diet.");
  }
  if (Number(data.fbs) === 1) {
    recs.push("Elevated fasting blood sugar may increase cardiovascular risk. Monitor glucose levels.");
    lifestyle.push("Limit refined sugars and maintain consistent meal timing.");
  }
  if (Number(data.exang) === 1) {
    recs.push("Exercise-induced angina requires immediate medical evaluation.");
    lifestyle.push("Avoid strenuous activity until cleared by a cardiologist.");
  }
  if (Number(data.thalach) < 120) {
    recs.push("Low max heart rate during exertion may indicate reduced cardiac capacity.");
  }
  if (Number(data.cp) >= 1 && Number(data.cp) <= 2) {
    recs.push("Chest pain symptoms warrant a thorough cardiac workup.");
  }

  if (isHighRisk) {
    recs.push("Based on model output, you are at elevated risk. Please consult a cardiologist promptly.");
    lifestyle.push("Begin light aerobic activity (walking 20–30 min/day) if approved by your doctor.");
    lifestyle.push("Practice stress management: meditation, deep breathing, or yoga.");
    lifestyle.push("Ensure 7–8 hours of quality sleep each night.");
  } else {
    recs.push("Your current profile suggests lower risk. Maintain healthy habits and regular screenings.");
    lifestyle.push("Stay active with 150 minutes of moderate exercise per week.");
    lifestyle.push("Maintain a heart-healthy Mediterranean-style diet.");
    lifestyle.push("Stay hydrated and limit alcohol consumption.");
  }

  return {
    recommendations: [...new Set(recs)],
    lifestyle: [...new Set(lifestyle)],
  };
}

// ─── Form Handler ────────────────────────────────────────────────
const form = document.getElementById("predictionForm");
const resultsPanel = document.getElementById("resultsPanel");
let riskChart;

form.addEventListener("submit", async (e) => {
  e.preventDefault();

  const formData = new FormData(form);
  const data = Object.fromEntries(formData.entries());

  // Validate all fields filled
  for (const value of Object.values(data)) {
    if (value === "") {
      alert("Please fill in all fields.");
      return;
    }
  }

  const { probability, isHighRisk } = await getPrediction(data);
  const pct = Math.round(probability * 100);

  // Update UI
  resultsPanel.classList.remove("hidden");

  const badge = document.getElementById("riskBadge");
  badge.textContent = isHighRisk ? "High Risk" : "Low Risk";
  badge.className = `risk-badge risk-badge--${isHighRisk ? "high" : "low"}`;

  document.getElementById("riskMessage").textContent = isHighRisk
    ? "Our logistic regression model indicates elevated heart disease risk. Please seek professional medical advice."
    : "Our logistic regression model indicates lower heart disease risk based on your inputs. Keep maintaining healthy habits!";

  document.getElementById("probabilityFill").style.width = `${pct}%`;
  document.getElementById("probabilityText").textContent = `${pct}%`;

  // Recommendations + lifestyle tips (rendered before the chart so a
  // chart/library failure can never block them from showing)
  const { recommendations, lifestyle } = getRecommendations(data, isHighRisk);

  document.getElementById("recommendationsList").innerHTML = recommendations
    .map((r) => `<li>${r}</li>`)
    .join("");

  document.getElementById("lifestyleList").innerHTML = lifestyle
    .map((l) => `<li>${l}</li>`)
    .join("");

  // Pie chart (best-effort — wrapped so a Chart.js load failure
  // doesn't take down the rest of the results panel)
  try {
    const ctx = document.getElementById("riskChart").getContext("2d");
    if (riskChart) {
      riskChart.destroy();
    }
    riskChart = new Chart(ctx, {
      type: "pie",
      data: {
        labels: ["Risk", "Safe"],
        datasets: [
          {
            data: [pct, 100 - pct],
            backgroundColor: ["#E8957A", "#7AAB8F"],
            borderWidth: 0,
          },
        ],
      },
      options: {
        responsive: true,
        plugins: {
          legend: { position: "bottom" },
        },
      },
    });
  } catch (err) {
    console.warn("Risk chart could not be rendered:", err);
  }

  resultsPanel.scrollIntoView({ behavior: "smooth", block: "nearest" });
});

// ─── Medicine Reminders (localStorage) ───────────────────────────
const reminderForm = document.getElementById("reminderForm");
const reminderList = document.getElementById("reminderList");

function loadReminders() {
  return JSON.parse(localStorage.getItem("heartcare_reminders") || "[]");
}

function saveReminders(reminders) {
  localStorage.setItem("heartcare_reminders", JSON.stringify(reminders));
}

function renderReminders() {
  const reminders = loadReminders();
  reminderList.innerHTML = reminders.length
    ? reminders
        .map(
          (r, i) => `
        <li>
          <span><strong>${r.name}</strong> — ${r.time}</span>
          <button onclick="deleteReminder(${i})">Remove</button>
        </li>`
        )
        .join("")
    : "<li style='color:#8b6f5c'>No reminders yet. Add one above.</li>";
}

window.deleteReminder = function (index) {
  const reminders = loadReminders();
  reminders.splice(index, 1);
  saveReminders(reminders);
  renderReminders();
};

reminderForm.addEventListener("submit", (e) => {
  e.preventDefault();
  const name = document.getElementById("medicineName").value.trim();
  const time = document.getElementById("reminderTime").value;
  if (!name || !time) return;

  const reminders = loadReminders();
  reminders.push({ name, time });
  saveReminders(reminders);
  renderReminders();
  reminderForm.reset();
});

// Check reminders every minute
function checkReminders() {
  const now = new Date();
  const currentTime = `${String(now.getHours()).padStart(2, "0")}:${String(now.getMinutes()).padStart(2, "0")}`;
  loadReminders().forEach((r) => {
    if (r.time === currentTime && !r.notified) {
      if ("Notification" in window && Notification.permission === "granted") {
        new Notification("HeartCare Reminder", {
          body: `Time to take your ${r.name}`,
          icon: "♥",
        });
      }
    }
  });
}

if ("Notification" in window && Notification.permission === "default") {
  Notification.requestPermission();
}

setInterval(checkReminders, 60000);
renderReminders();
function sendMessage() {
  const input = document.getElementById("chat-input");
  const messages = document.getElementById("chat-messages");

  const text = input.value.trim();
  if (!text) return;

  messages.innerHTML += `<div><b>You:</b> ${text}</div>`;

  let reply = "Please consult a healthcare professional.";

  if (text.toLowerCase().includes("cholesterol")) {
    reply = "High cholesterol can increase heart disease risk. Regular exercise and a healthy diet help.";
  }
  else if (text.toLowerCase().includes("blood pressure")) {
    reply = "Maintaining blood pressure below 120/80 mmHg is generally recommended.";
  }
  else if (text.toLowerCase().includes("heart")) {
    reply = "Heart health improves with exercise, healthy eating, good sleep, and stress management.";
  }

  messages.innerHTML += `<div class="bot-msg">${reply}</div>`;

  input.value = "";
  messages.scrollTop = messages.scrollHeight;
}