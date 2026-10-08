import { NavLink, Route, Routes } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "./lib/api";
import Dashboard from "./pages/Dashboard";
import Politicians from "./pages/Politicians";
import Signals from "./pages/Signals";
import SettingsPage from "./pages/Settings";

function KillSwitch() {
  const qc = useQueryClient();
  const { data: status } = useQuery({ queryKey: ["status"], queryFn: api.status });
  const kill = useMutation({
    mutationFn: api.kill,
    onSuccess: () => qc.invalidateQueries(),
  });
  const mode = status?.trading_mode ?? "DISABLED";
  return (
    <div className="row" style={{ marginTop: 24, flexDirection: "column", alignItems: "stretch", gap: 10 }}>
      <div className="row spread">
        <span className="muted" style={{ fontSize: 13 }}>Mode</span>
        <span className={`mode-pill mode-${mode}`}>{mode}</span>
      </div>
      <button
        className="btn danger small"
        onClick={() => {
          if (confirm("Cancel all open orders and disable trading?")) kill.mutate();
        }}
      >
        ■ Kill switch
      </button>
    </div>
  );
}

export default function App() {
  return (
    <div className="layout">
      <aside className="sidebar">
        <div className="brand">
          📈 Portfolio Tracker
          <small>politicians + your copy trades</small>
        </div>
        <nav className="nav">
          <NavLink to="/" end className={({ isActive }) => (isActive ? "active" : "")}>
            Dashboard
          </NavLink>
          <NavLink to="/politicians" className={({ isActive }) => (isActive ? "active" : "")}>
            Politicians
          </NavLink>
          <NavLink to="/signals" className={({ isActive }) => (isActive ? "active" : "")}>
            Copy Signals
          </NavLink>
          <NavLink to="/settings" className={({ isActive }) => (isActive ? "active" : "")}>
            Settings
          </NavLink>
        </nav>
        <KillSwitch />
      </aside>
      <main className="main">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/politicians" element={<Politicians />} />
          <Route path="/signals" element={<Signals />} />
          <Route path="/settings" element={<SettingsPage />} />
        </Routes>
      </main>
    </div>
  );
}
