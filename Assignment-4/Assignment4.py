import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import joblib # Using joblib to load the .pkl file
from sklearn.metrics import mean_squared_error
import imageio
import os
import tempfile
import math

# --- 1. OPTIMIZATION LOGIC (For Ground Truth Comparison) ---
def get_optimized_trajectory(q_start, q_end):
    T, N = 5.0, 50
    dt = T / (N - 1)
    time = np.linspace(0, T, N)

    def cubic_trajectory(q0, qf, T, t):
        return q0 + (3*(qf - q0)/T**2)*t**2 + (-2*(qf - q0)/T**3)*t**3

    x0 = np.concatenate([cubic_trajectory(q_start[0], q_end[0], T, time), 
                         cubic_trajectory(q_start[1], q_end[1], T, time)])

    def cost_function(x):
        q1_ddot = np.diff(x[:N], n=2) / dt**2
        q2_ddot = np.diff(x[N:], n=2) / dt**2
        return np.sum(q1_ddot**2 + q2_ddot**2)

    constraints = [
        {'type': 'eq', 'fun': lambda x: x[0] - q_start[0]},
        {'type': 'eq', 'fun': lambda x: x[N] - q_start[1]},
        {'type': 'eq', 'fun': lambda x: x[N-1] - q_end[0]},
        {'type': 'eq', 'fun': lambda x: x[2*N-1] - q_end[1]},
        {'type': 'eq', 'fun': lambda x: x[1] - x[0]},
        {'type': 'eq', 'fun': lambda x: x[N+1] - x[N]},
        {'type': 'eq', 'fun': lambda x: x[N-1] - x[N-2]},
        {'type': 'eq', 'fun': lambda x: x[2*N-1] - x[2*N-2]}
    ]

    res = minimize(cost_function, x0, method='SLSQP', constraints=constraints)
    return res.x

# --- 4. VIDEO / PLOTTING HELPERS ---
def _plot_arm(fig, q1, q2):
    ax = fig.add_subplot(1,1,1)
    ax.clear()
    # link lengths = 1, base at (0,0)
    x1, y1 = 0.0, 0.0
    x_e1 = x1 + math.cos(q1)
    y_e1 = y1 + math.sin(q1)
    x_e2 = x_e1 + math.cos(q1 + q2)
    y_e2 = y_e1 + math.sin(q1 + q2)

    ax.plot([x1, x_e1], [y1, y_e1], linewidth=2, color='tab:blue')
    ax.plot([x_e1, x_e2], [y_e1, y_e2], linewidth=2, color='tab:orange')
    ax.scatter([x1, x_e1, x_e2], [y1, y_e1, y_e2], color='k')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('2-Link Robotic Arm')
    ax.set_aspect('equal', 'box')
    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(-2.2, 2.2)

def create_video_from_traj(traj, out_path, N=50, fps=10):
    # traj: concatenated [q1(t0..), q2(t0..)] in radians
    tmp_dir = tempfile.mkdtemp(prefix='frames_')
    q1 = traj[:N]
    q2 = traj[N:]
    try:
        for i in range(N):
            fig = plt.figure(figsize=(5.12,5.12))
            _plot_arm(fig, q1[i], q2[i])
            frame_path = os.path.join(tmp_dir, f"{i:03d}.png")
            fig.savefig(frame_path, dpi=100)
            plt.close(fig)

        # write mp4 using imageio
        with imageio.get_writer(out_path, fps=fps) as writer:
            for i in range(N):
                frame_path = os.path.join(tmp_dir, f"{i:03d}.png")
                image = imageio.imread(frame_path)
                writer.append_data(image)
    finally:
        # cleanup frames
        try:
            for f in os.listdir(tmp_dir):
                os.remove(os.path.join(tmp_dir, f))
            os.rmdir(tmp_dir)
        except Exception:
            pass

# --- 2. LOAD PRE-TRAINED MODEL WEIGHTS ---
@st.cache_resource
def load_trained_model():
    try:
        # Loading your saved pickle file
        model = joblib.load('trajectory_model_weights.pkl')
        return model
    except FileNotFoundError:
        st.error("Error: 'trajectory_model_weights.pkl' not found.")
        return None

# --- 3. STREAMLIT UI ---
st.title("Assignment 4: Prediction and Dashboard")
st.sidebar.header("Input Joint Configurations (Degrees)")

q1_s_deg = st.sidebar.slider("Start Joint 1 (°)", -180.0, 180.0, 0.0)
q2_s_deg = st.sidebar.slider("Start Joint 2 (°)", -180.0, 180.0, 0.0)
q1_e_deg = st.sidebar.slider("End Joint 1 (°)", -180.0, 180.0, 90.0)
q2_e_deg = st.sidebar.slider("End Joint 2 (°)", -180.0, 180.0, 45.0)

q_start = np.radians([q1_s_deg, q2_s_deg])
q_end = np.radians([q1_e_deg, q2_e_deg])

# Instant prediction using loaded weights
model = load_trained_model()

if model is not None:
    # Optimized trajectory (Calculated in real-time)
    opt_traj = get_optimized_trajectory(q_start, q_end)
    
    # Learned trajectory (Predicted instantly using MLP weights) [cite: 19, 27]
    input_data = np.concatenate([q_start, q_end]).reshape(1, -1)
    nn_traj = model.predict(input_data)[0]

    # Visual Comparison Plots 
    t = np.linspace(0, 5, 50)
    opt_traj_deg = np.degrees(opt_traj)
    nn_traj_deg = np.degrees(nn_traj)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Joint 1 Path")
        fig1, ax1 = plt.subplots()
        ax1.plot(t, opt_traj_deg[:50], 'g', label="Optimized")
        ax1.plot(t, nn_traj_deg[:50], 'r--', label="NN Predicted")
        ax1.set_xlabel("Time (s)"); ax1.set_ylabel("Angle (°)"); ax1.legend()
        st.pyplot(fig1)

    with col2:
        st.subheader("Joint 2 Path")
        fig2, ax2 = plt.subplots()
        ax2.plot(t, opt_traj_deg[50:], 'g', label="Optimized")
        ax2.plot(t, nn_traj_deg[50:], 'r--', label="NN Predicted")
        ax2.set_xlabel("Time (s)"); ax2.set_ylabel("Angle (°)"); ax2.legend()
        st.pyplot(fig2)

    # MSE Evaluation 
    mse = mean_squared_error(opt_traj_deg, nn_traj_deg)
    st.write(f"**Current Prediction Mean Squared Error:** {mse:.6f} sq. degrees")

    # --- Video generation and display ---
    if st.checkbox("Show arm videos (predicted and optimized)"):
        try:
            tmpdir = tempfile.mkdtemp(prefix='videos_')
            pred_path = os.path.join(tmpdir, "robot_arm_pred.mp4")
            true_path = os.path.join(tmpdir, "robot_arm_true.mp4")
            with st.spinner("Generating videos..."):
                create_video_from_traj(nn_traj, pred_path, N=50, fps=10)
                create_video_from_traj(opt_traj, true_path, N=50, fps=10)

            st.subheader("NN Predicted Trajectory")
            st.video(pred_path)
            with open(pred_path, "rb") as f:
                st.download_button("Download predicted video", f.read(), file_name="robot_arm_pred.mp4")

            st.subheader("Optimized (Ground-truth) Trajectory")
            st.video(true_path)
            with open(true_path, "rb") as f:
                st.download_button("Download optimized video", f.read(), file_name="robot_arm_true.mp4")

            # cleanup video files (keep temp dir removal safe)
            try:
                os.remove(pred_path)
                os.remove(true_path)
                os.rmdir(tmpdir)
            except Exception:
                pass
        except Exception as e:
            st.error(f"Video generation failed: {e}")