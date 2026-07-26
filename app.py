from flask import Flask, request, jsonify, send_from_directory
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os

app = Flask(__name__)

# Load the saved XGBoost model
model = pickle.load(open("xgboost_model.pkl", "rb"))

# ── Feature options (same as training data) ───────────────────────────
COMPANIES   = ["Acer", "Apple", "Asus", "Chuwi", "Dell", "Fujitsu",
               "Google", "HP", "Huawei", "LG", "Lenovo", "MSI",
               "Mediacom", "Microsoft", "Razer", "Samsung",
               "Toshiba", "Vero", "Xiaomi"]

TYPE_NAMES  = ["2 in 1 Convertible", "Gaming", "Netbook",
               "Notebook", "Ultrabook", "Workstation"]

SCREEN_RES  = ["1366x768", "1440x900", "1600x900", "1920x1080",
               "1920x1200", "2160x1440", "2256x1504", "2304x1440",
               "2560x1440", "2560x1600", "2880x1800", "3200x1800",
               "3840x2160"]

CPU_OPTIONS = ["Intel Core i3", "Intel Core i5", "Intel Core i7",
               "Intel Core M", "AMD A4", "AMD A6", "AMD A9",
               "AMD E-Series", "AMD Ryzen 3", "AMD Ryzen 5",
               "AMD Ryzen 7", "Samsung", "Intel Atom", "Intel Celeron",
               "Intel Pentium", "Intel Xeon"]

MEMORY_OPTS = ["16GB", "32GB", "4GB", "64GB", "8GB"]
HDD_OPTS    = ["0GB", "1.0TB", "1TB", "2.0TB", "500GB"]
GPU_OPTS    = ["AMD", "ARM", "Intel", "Nvidia"]
OP_SYS      = ["Chrome OS", "Linux", "No OS", "Windows 10",
               "Windows 10 S", "Windows 7", "macOS", "Android"]

BRAND_IMAGES = {
    "Apple":     "https://images.unsplash.com/photo-1517336714731-489689fd1ca8?w=600&q=80",
    "Dell":      "https://images.unsplash.com/photo-1593642702821-c8da6771f0c6?w=600&q=80",
    "HP":        "https://images.unsplash.com/photo-1496181133206-80ce9b88a853?w=600&q=80",
    "Lenovo":    "https://images.unsplash.com/photo-1588872657578-7efd1f1555ed?w=600&q=80",
    "Asus":      "https://images.unsplash.com/photo-1603302576837-37561b2e2302?w=600&q=80",
    "Acer":      "https://images.unsplash.com/photo-1525547719571-a2d4ac8945e2?w=600&q=80",
    "MSI":       "https://images.unsplash.com/photo-1611532736597-de2d4265fba3?w=600&q=80",
    "Microsoft": "https://images.unsplash.com/photo-1542393545-10f5cde2c810?w=600&q=80",
    "Samsung":   "https://images.unsplash.com/photo-1611186871348-b1ce696e52c9?w=600&q=80",
    "Razer":     "https://images.unsplash.com/photo-1593642632559-0c6d3fc62b89?w=600&q=80",
}
DEFAULT_IMAGE = "https://images.unsplash.com/photo-1531297484001-80022131f5a1?w=600&q=80"


def label_encode(value, options):
    le = LabelEncoder()
    le.fit(sorted(options))
    return int(le.transform([value])[0])


# ── Serve index.html directly from the same folder as app.py ──────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")


# ── Predict endpoint ──────────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        company    = data["company"]
        type_name  = data["type_name"]
        inches     = float(data["inches"])
        screen_res = data["screen_res"]
        cpu        = data["cpu"]
        ram        = int(data["ram"])
        memory     = data["memory"]
        gpu        = data["gpu"]
        op_sys     = data["op_sys"]
        weight     = float(data["weight"])

        company_enc = label_encode(company,    COMPANIES)
        type_enc    = label_encode(type_name,  TYPE_NAMES)
        screen_enc  = label_encode(screen_res, SCREEN_RES)
        cpu_enc     = label_encode(cpu,        CPU_OPTIONS)
        memory_enc  = label_encode(memory,     MEMORY_OPTS)
        hdd_enc     = label_encode("0GB",      HDD_OPTS)   # fixed default
        gpu_enc     = label_encode(gpu,        GPU_OPTS)
        os_enc      = label_encode(op_sys,     OP_SYS)

        # 11 features: Company, TypeName, Inches, ScreenResolution,
        #              Cpu, Ram, Memory, HDD, Gpu, OpSys, Weight
        features = np.array([[company_enc, type_enc, inches, screen_enc,
                               cpu_enc, ram, memory_enc, hdd_enc,
                               gpu_enc, os_enc, weight]])

        price     = float(model.predict(features)[0])
        image_url = BRAND_IMAGES.get(company, DEFAULT_IMAGE)

        return jsonify({
            "price":     round(price, 2),
            "image_url": image_url,
            "specs": {
                "Brand":   company,
                "Type":    type_name,
                "Screen":  f'{inches}" — {screen_res}',
                "CPU":     cpu,
                "RAM":     f"{ram} GB",
                "Storage": memory,
                "GPU":     gpu,
                "OS":      op_sys,
                "Weight":  f"{weight} kg"
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True)