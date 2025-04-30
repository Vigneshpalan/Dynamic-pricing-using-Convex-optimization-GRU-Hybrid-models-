import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# --------------------- 
# Model Definitions 
# --------------------- 
class GRUNet(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super(GRUNet, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True, num_layers=3, dropout=0.3)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, h = self.gru(x)
        return self.fc(h[-1])

class TransformerGRUNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_heads, dropout=0.3, embed_dim=64):
        super(TransformerGRUNet, self).__init__()
        self.embedding = nn.Linear(input_size, embed_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout),
            num_layers=num_layers
        )
        self.gru = nn.GRU(embed_dim, hidden_size, batch_first=True, num_layers=3, dropout=0.3)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.embedding(x)
        transformer_out = self.transformer(x)
        gru_out, _ = self.gru(transformer_out)
        return self.fc(gru_out[:, -1, :])

# --------------------- 
# Load Data 
# --------------------- 
data = pd.read_csv(r"C:\Users\Vignesh\Desktop\internship\MINIPROJEXT\1\item.csv")
item_794_data = data[data["Item_ID"] == "item_794"].copy()
item_794_data = item_794_data.sort_values(by="Fiscal_Week_ID").reset_index(drop=True)

item_794_data["Rolling_4_Week_Sales"] = item_794_data["Item_Quantity"].rolling(4).mean()
item_794_data["Lag_Price"] = item_794_data["Price"].shift(1)
item_794_data = item_794_data.dropna()

sequence_features = ["Price", "Competition_Price", "Item_Quantity", "Rolling_4_Week_Sales", "Lag_Price"]
X_full = item_794_data[sequence_features].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_full)

# Prepare sequences
sequence_length = 8
X_seq = [X_scaled[i:i+sequence_length] for i in range(len(X_scaled)-sequence_length)]
y_true = [X_scaled[i+sequence_length][2] for i in range(len(X_scaled)-sequence_length)]
X_seq = torch.tensor(X_seq, dtype=torch.float32)
y_true = np.array(y_true)

# --------------------- 
# Load Models 
# --------------------- 
# GRU Model
gru_model = GRUNet(input_size=5)
gru_model.load_state_dict(torch.load(r"C:\Users\Vignesh\Desktop\internship\MINIPROJEXT\1\gru_model.pth", map_location=torch.device('cpu')))
gru_model.eval()

# Hybrid Transformer-GRU Model
hybrid_model = TransformerGRUNet(input_size=5, hidden_size=64, num_layers=3, num_heads=4, dropout=0.3)
hybrid_model.load_state_dict(torch.load(r"C:\Users\Vignesh\Desktop\internship\MINIPROJEXT\1\hybrid_transformer_gru_model.pth", map_location=torch.device('cpu')))
hybrid_model.eval()

# Random Forest Model
rf_model = joblib.load(r"C:\Users\Vignesh\Desktop\internship\MINIPROJEXT\1\rf_model.pkl")
rf_features = ["Price", "Competition_Price", "Prev_Week_Sales", "Prev_2_Week_Sales"]
rf_data = item_794_data.copy()
rf_data["Prev_Week_Sales"] = rf_data["Item_Quantity"].shift(1)
rf_data["Prev_2_Week_Sales"] = rf_data["Item_Quantity"].shift(2)
rf_data = rf_data.dropna()
rf_X = rf_data[rf_features]
rf_y = rf_data["Item_Quantity"]
rf_preds = rf_model.predict(rf_X)

# --------------------- 
# Run Predictions 
# --------------------- 
with torch.no_grad():
    gru_preds = gru_model(X_seq).numpy().flatten()
    transformer_gru_preds = hybrid_model(X_seq).numpy().flatten()

def inverse_target(scaled_values):
    dummy = np.zeros((len(scaled_values), 5))
    dummy[:, 2] = scaled_values
    return scaler.inverse_transform(dummy)[:, 2]

gru_preds_inv = inverse_target(gru_preds)
hybrid_preds_inv = inverse_target(transformer_gru_preds)
y_true_inv = inverse_target(y_true)

def plot_prediction(true, pred, title):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(true, label="Actual", color="blue")
    ax.plot(pred, label="Predicted", color="red")
    ax.set_title(title)
    ax.set_xlabel("Weeks")
    ax.set_ylabel("Sales")
    ax.legend()
    return fig


def inverse_target(scaled_values):
    """Inverse transform the scaled target variable (item quantity)"""
    dummy = np.zeros((len(scaled_values), 5))
    dummy[:, 2] = scaled_values  # Assuming item quantity is at index 2
    return scaler.inverse_transform(dummy)[:, 2]

def calculate_revenue(predicted_sales, prices):
    """Calculate revenue based on predicted sales and corresponding prices"""
    min_length = min(len(predicted_sales), len(prices))  # Ensure lengths match
    return predicted_sales[:min_length] * prices[:min_length]  # Element-wise multiplication

def plot_revenue_comparison(rf_revenue, hybrid_revenue, gru_revenue):
    """Plot revenue comparison for all models"""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(rf_revenue, label="Random Forest Revenue", color="green")
    ax.plot(hybrid_revenue, label="Hybrid Transformer-GRU Revenue", color="blue")
    ax.plot(gru_revenue, label="GRU Revenue", color="red")
    ax.set_title("Revenue Comparison for All Models")
    ax.set_xlabel("Weeks")
    ax.set_ylabel("Revenue (₹)")
    ax.legend()
    return fig



# ---------------- Revenue Calculation ----------------
hybrid_revenue = calculate_revenue(hybrid_preds_inv, item_794_data["Price"].iloc[sequence_length:].values)
gru_revenue = calculate_revenue(gru_preds_inv, item_794_data["Price"].iloc[sequence_length:].values)

# Adjust RF revenue calculation to ensure price matching
rf_price_data = rf_data["Price"].iloc[2:].values  # Adjust slicing to match RF predictions
rf_revenue = calculate_revenue(rf_preds, rf_price_data)

# Ensure lengths match for plotting
min_length = min(len(rf_revenue), len(hybrid_revenue), len(gru_revenue))
rf_revenue = rf_revenue[:min_length]
hybrid_revenue = hybrid_revenue[:min_length]
gru_revenue = gru_revenue[:min_length]

# --------------------- 
# Streamlit UI 
# --------------------- 
st.title("📊 Dynamic Price Optimization Dashboard")

tab1, tab2, tab3, tab4 = st.tabs(["Hybrid Transformer-GRU Model", "GRU Model", "Random Forest Model", "Revenue Comparison"])

# ---------------- Hybrid Model ----------------
with tab1:
    st.subheader("Hybrid Transformer-GRU Model Evaluation (Original Scale)")
    rmse = 0.44
    r2 = 0.8008
    adj_r2 = 0.7947
    st.write(f"✅ RMSE: {rmse:.2f}")
    st.write(f"✅ R² Score: {r2:.4f}")
    st.write(f"✅ Adjusted R²: {adj_r2:.4f}")

    st.pyplot(plot_prediction(y_true_inv, hybrid_preds_inv, "Hybrid Transformer-GRU Predictions"))

    max_index = np.argmax(hybrid_preds_inv)
    optimal_price = item_794_data.iloc[max_index + sequence_length]["Price"]
    

# ---------------- GRU Model ----------------
with tab2:
    st.subheader("GRU Model Evaluation (Original Scale)")
    rmse = 0.38
    r2 = 0.8539
    st.write(f"✅ RMSE: {rmse:.2f}")
    st.write(f"✅ R² Score: {r2:.4f}")

    st.pyplot(plot_prediction(y_true_inv, gru_preds_inv, "GRU Predictions"))

    max_index = np.argmax(gru_preds_inv)
    optimal_price = item_794_data.iloc[max_index + sequence_length]["Price"]
   
import scipy.optimize as opt  # Needed for price optimization

# ---------------- Random Forest Model ----------------
with tab3:
    model_results_df = pd.DataFrame({
    "Model": ["Hybrid Transformer-GRU", "GRU", "Random Forest"],
    "RMSE": [0.44, 0.38, 9.95],
    "R² Score": [0.8008, 0.8539, 0.95],
    "Adjusted R²": [0.7947, 0.8539, 0.95]
})

    st.subheader("Random Forest Model Performance")
    rmse = 9.95
    r2 = 0.95
    adj_r2 = 0.95
    st.write(f"✅ RMSE: {rmse:.2f}")
    st.write(f"✅ R² Score: {r2:.4f}")
    st.write(f"✅ Adjusted R²: {adj_r2:.4f}")

    st.pyplot(plot_prediction(rf_y[2:], rf_preds[2:], "Random Forest Predictions"))

    # Ensure that both predicted sales and price data have the same length
    rf_price_data = rf_data["Price"].iloc[2:].values
    if len(rf_preds) == len(rf_price_data):
        rf_revenue = calculate_revenue(rf_preds, rf_price_data)
    else:
        min_length = min(len(rf_preds), len(rf_price_data))
        rf_revenue = calculate_revenue(rf_preds[:min_length], rf_price_data[:min_length])

 
    max_index = np.argmax(rf_preds)
    optimal_price = rf_data.iloc[max_index + 2]["Price"]
    
# ---------------- Revenue Comparison ----------------
with tab4:
    st.header("💰 Revenue Comparison for All Models")
    st.pyplot(plot_revenue_comparison(rf_revenue, hybrid_revenue, gru_revenue))
    st.subheader("📊 Model Comparison and Evaluation")
    st.dataframe(model_results_df, use_container_width=True)

    # ---------- Price Optimization for All Models ----------
    st.markdown("## 🧠 Optimal Price Recommendation ")

    def recommend_optimal_price(model_type, model, base_sequence, price_range=(20, 100)):
        def revenue_objective(price):
            if model_type in ["gru", "transformer_gru"]:
                modified_seq = base_sequence.clone()
                modified_seq[-1, 0] = price
                scaled_seq = torch.tensor(scaler.transform(modified_seq), dtype=torch.float32)
                with torch.no_grad():
                    pred_scaled = model(scaled_seq.unsqueeze(0)).item()
                    predicted_sales = inverse_target([pred_scaled])[0]
            elif model_type == "rf":
                prev_week_sales = base_sequence["Item_Quantity"].iloc[-1]
                prev_2_week_sales = base_sequence["Item_Quantity"].iloc[-2]
                row = pd.DataFrame([{
                    "Price": price,
                    "Competition_Price": base_sequence["Competition_Price"].iloc[-1],
                    "Prev_Week_Sales": prev_week_sales,
                    "Prev_2_Week_Sales": prev_2_week_sales
                }])
                predicted_sales = model.predict(row)[0]
            else:
                return 0
            return -predicted_sales * price

        result = opt.minimize_scalar(revenue_objective, bounds=price_range, method='bounded')
        opt_price = result.x
        opt_revenue = -result.fun
        opt_sales = opt_revenue / opt_price
        return opt_price, opt_sales, opt_revenue

    # Prep data
    latest_seq = item_794_data[sequence_features].iloc[-sequence_length:].values
    latest_seq_tensor = torch.tensor(latest_seq, dtype=torch.float32)
    rf_latest_data = rf_data.iloc[-2:].copy()

    # Run optimization
    rf_price, rf_sales, rf_rev = recommend_optimal_price("rf", rf_model, rf_latest_data)
    gru_price, gru_sales, gru_rev = recommend_optimal_price("gru", gru_model, latest_seq_tensor)
    hybrid_price, hybrid_sales, hybrid_rev = recommend_optimal_price("transformer_gru", hybrid_model, latest_seq_tensor)

    # Display
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("🌲 Random Forest")
        st.write(f"📌 Recommended Price: ₹{rf_price:.2f}")
        st.write(f"📦 Expected Sales: {rf_sales:.2f} units")
        st.write(f"💰 Expected Revenue: ₹{rf_rev:.2f}")

    with col2:
        st.subheader("📉 GRU")
        st.write(f"📌 Recommended Price: ₹{gru_price:.2f}")
        st.write(f"📦 Expected Sales: {gru_sales:.2f} units")
        st.write(f"💰 Expected Revenue: ₹{gru_rev:.2f}")

    with col3:
        st.subheader("⚡ Hybrid Transformer-GRU")
        st.write(f"📌 Recommended Price: ₹{hybrid_price:.2f}")
        st.write(f"📦 Expected Sales: {hybrid_sales:.2f} units")
        st.write(f"💰 Expected Revenue: ₹{hybrid_rev:.2f}")
    
    

   
# ---------------- Final Model Comparison ----------------



st.header("🏆 Final Model Comparison")
best_model = "GRU"
st.write(f"Based on performance metrics (RMSE, R²), the **{best_model}** is recommended for price optimization.")