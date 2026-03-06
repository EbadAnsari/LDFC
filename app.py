from flask import Flask, render_template_string
import json

app = Flask(__name__)

LOG_FILE = "log.json"

def read_log():
    with open(LOG_FILE, "r") as f:
        return json.load(f)

@app.route("/")
def dashboard():
    data = read_log()
    return render_template_string("""
<!DOCTYPE html>
<html>
<head>
  <title>Training Dashboard</title>
  <style>
    body { font-family: Arial; padding: 20px; }
    .card { border: 1px solid #ccc; padding: 15px; width: 320px; }
  </style>
</head>
<body>

<h2>📊 Training Status</h2>

<div class="card">
  <p><b>Epoch:</b> {{ data.epoch }}</p>
  <p><b>Train Loss:</b> {{ data.train_loss }}</p>
  <p><b>Val Loss:</b> {{ data.val_loss }}</p>
  <p><b>Elapsed Time:</b> <span id="elapsed"></span></p>
</div>

<script>
  // Start time from server (ONE TIME)
  const startTime = new Date("{{ data.time }}".replace(" ", "T"));

  function updateElapsed() {
    const now = new Date();
    const diff = now - startTime;

    const s = Math.floor(diff / 1000);
    const h = Math.floor(s / 3600);
    const m = Math.floor((s % 3600) / 60);

    document.getElementById("elapsed").innerText =
      `${h}h ${m}m ${s % 60}s`;
  }

  updateElapsed();
  setInterval(updateElapsed, 1000);
</script>

</body>
</html>
""", data=data)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)