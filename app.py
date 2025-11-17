from flask import Flask, render_template, request, redirect, url_for, session, flash
import mysql.connector
import joblib
import pandas as pd
from datetime import datetime

app = Flask(__name__)
app.secret_key = "change-this-in-prod"  # used for session and flash

# --- Model and App Configuration ---
MODEL_PATH = "data/model_output/groundwater_model.joblib"
model = joblib.load(MODEL_PATH)

# MySQL credentials from create.md
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "",
    "database": "underwater_2025",
}



@app.after_request
def add_header(response):

  response.cache_control.no_store = True
  return response





def get_db_connection(db_required: bool = True):
    """Create a new MySQL connection. If db_required is False, omit database selection."""
    cfg = DB_CONFIG.copy()
    if not db_required:
        cfg.pop("database", None)
    return mysql.connector.connect(**cfg)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")  # as per spec, no hashing

        if not name or not email or not password:
            flash("All fields are required.", "danger")
            return render_template("register.html")

        try:
            conn = get_db_connection()
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO users (name, email, password)
                VALUES (%s, %s, %s)
                """,
                (name, email, password),
            )
            conn.commit()
            flash("Registration successful. Please login.", "success")
            return redirect(url_for("login"))
        except mysql.connector.IntegrityError:
            flash("Email already registered.", "warning")
            return render_template("register.html")
        except Exception as e:
            flash(f"Error: {e}", "danger")
            return render_template("register.html")
        finally:
            try:
                cur.close()
                conn.close()
            except Exception:
                pass

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        if not email or not password:
            flash("Email and password are required.", "danger")
            return render_template("login.html")
        try:
            conn = get_db_connection()
            cur = conn.cursor(dictionary=True)
            cur.execute(
                "SELECT id, name, email FROM users WHERE email=%s AND password=%s",
                (email, password)
            )
            user = cur.fetchone()
            cur.close()
            conn.close()
            if user:
                session["user_id"] = user["id"]
                session["user_name"] = user["name"]
                session["user_email"] = user["email"]
                flash("Login successful.", "success")
                return redirect(url_for("predict"))
            else:
                flash("Invalid email or password.", "danger")
                return render_template("login.html")
        except Exception as e:
            flash(f"Error: {e}", "danger")
            return render_template("login.html")
    return render_template("login.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if not session.get("user_id"):
        flash("Please login to access Predict.", "info")
        return redirect(url_for("login"))

    if request.method == "POST":
        try:
            latitude = float(request.form["latitude"])
            longitude = float(request.form["longitude"])
            date_str = request.form["date"]

            # --- Feature Engineering for Prediction ---
            prediction_date = datetime.strptime(date_str, "%Y-%m-%d")
            new_data = pd.DataFrame({
                'LATITUDE': [latitude],
                'LONGITUDE': [longitude],
                'DATE': [pd.to_datetime(prediction_date)]
            })
            new_data['YEAR'] = new_data['DATE'].dt.year
            new_data['MONTH'] = new_data['DATE'].dt.month
            new_data['DAY_OF_YEAR'] = new_data['DATE'].dt.dayofyear
            new_data['LAT_LON_PRODUCT'] = new_data['LATITUDE'] * new_data['LONGITUDE']

            prediction_features = new_data[['LATITUDE', 'LONGITUDE', 'YEAR', 'MONTH', 'DAY_OF_YEAR', 'LAT_LON_PRODUCT']]

            # --- Make Prediction ---
            predicted_dtwl = model.predict(prediction_features)[0]

            # --- Generate Suggestion ---
            suggestion = groundwater_suggestion(predicted_dtwl)

            # --- Save to Database ---
            conn = get_db_connection()
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO predictions (user_id, latitude, longitude, prediction_date, predicted_dtwl)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (session["user_id"], latitude, longitude, prediction_date, predicted_dtwl)
            )
            conn.commit()
            cur.close()
            conn.close()

            # Store result and suggestion in session to pass to the result page
            session['prediction_result'] = {
                'latitude': latitude,
                'longitude': longitude,
                'date': date_str,
                'dtwl': f"{predicted_dtwl:.2f}",
                'suggestion': suggestion
            }
            return redirect(url_for("result"))
        except Exception as e:
            flash(f"An error occurred: {e}", "danger")
            return render_template("predict.html")
    return render_template("predict.html")

def groundwater_suggestion(dtwl):
    """Return a suggestion string based on the predicted groundwater level (DTWL)."""
    if dtwl < 5:
        return "Groundwater level is high. Water extraction is safe."
    elif 5 <= dtwl < 15:
        return "Groundwater level is moderate. Use water judiciously."
    else:
        return "Groundwater level is low. Limit extraction and consider conservation measures."


@app.route("/result")
def result():
    if not session.get("user_id"):
        return redirect(url_for("login"))
    
    prediction_result = session.pop('prediction_result', None)
    if not prediction_result:
        return redirect(url_for("predict"))

    # Get feature importance from model if available
    feature_names = ["LATITUDE", "LONGITUDE", "YEAR", "MONTH", "DAY_OF_YEAR", "LAT_LON_PRODUCT"]
    feature_importances = []
    if hasattr(model, 'feature_importances_'):
        feature_importances = list(model.feature_importances_)
    elif hasattr(model, 'coef_'):
        feature_importances = list(abs(model.coef_))
    # else: leave empty

    # Prepare history arrays for chart
    history_labels = []
    history_data = []
    try:
        # If you have history data available, populate these lists
        # For example, if you query predictions for the user:
        conn = get_db_connection()
        cur = conn.cursor(dictionary=True)
        cur.execute(
            "SELECT prediction_date, predicted_dtwl FROM predictions WHERE user_id = %s ORDER BY prediction_date ASC",
            (session["user_id"],)
        )
        rows = cur.fetchall()
        history_labels = [row["prediction_date"].strftime('%Y-%m-%d') if hasattr(row["prediction_date"], 'strftime') else str(row["prediction_date"]) for row in rows]
        history_data = [row["predicted_dtwl"] for row in rows]
        cur.close()
        conn.close()
    except Exception:
        history_labels = []
        history_data = []

    return render_template(
        "result.html",
        result=prediction_result,
        feature_names=feature_names or [],
        feature_importances=feature_importances or [],
        history_labels=history_labels or [],
        history_data=history_data or []
    )


@app.route("/history")
def history():
    if not session.get("user_id"):
        flash("Please login to view your history.", "info")
        return redirect(url_for("login"))

    try:
        conn = get_db_connection()
        cur = conn.cursor(dictionary=True)
        cur.execute(
            "SELECT latitude, longitude, prediction_date, predicted_dtwl, created_at FROM predictions WHERE user_id = %s ORDER BY created_at DESC",
            (session["user_id"],)
        )
        history_data = cur.fetchall()
        cur.close()
        conn.close()
        return render_template("history.html", history=history_data)
    except Exception as e:
        flash(f"Could not retrieve history: {e}", "danger")
        return redirect(url_for("predict"))


@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out successfully.", "success")
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=True)