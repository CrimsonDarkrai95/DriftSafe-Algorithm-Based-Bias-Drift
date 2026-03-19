from flask import Flask, render_template, jsonify, request
import usa_model
  # Import the model module to access get_fairness_metrics

app = Flask(__name__)

@app.route("/")
def home():
    return "DriftSafe backend is running."

@app.route("/india")
def india():
    return render_template("india.html")

@app.route("/test")
def test():
    return "TEST ROUTE WORKS"

@app.route("/usa")
def usa_page():
    print("USA ROUTE HIT")
    return render_template("usa.html")

@app.route("/api/usa")
def api_usa():
    dimension = request.args.get('dimension', 'age_group')
    return jsonify(usa_model.get_fairness_metrics(dimension))

if __name__ == "__main__":
    app.run(debug=True)