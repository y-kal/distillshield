import { useEffect, useState } from "react";
import { fetchJson, postJson } from "./api";

const emptyFilters = { label: "", policy: "", confidence: "" };

export default function App() {
  const [sessions, setSessions] = useState([]);
  const [selectedId, setSelectedId] = useState(null);
  const [detail, setDetail] = useState(null);
  const [models, setModels] = useState([]);
  const [experiment, setExperiment] = useState(null);
  const [filters, setFilters] = useState(emptyFilters);
  const [error, setError] = useState("");

  useEffect(() => {
    refresh();
  }, []);

  useEffect(() => {
    if (selectedId) {
      fetchJson(`/sessions/${selectedId}`).then(setDetail).catch((err) => setError(err.message));
    }
  }, [selectedId]);

  async function refresh() {
    try {
      const [sessionData, modelData, latestExperiment] = await Promise.all([
        fetchJson("/sessions"),
        fetchJson("/models"),
        fetchJson("/experiments/latest"),
      ]);
      setSessions(sessionData);
      setModels(modelData);
      setExperiment(latestExperiment);
      if (sessionData.length && !selectedId) {
        setSelectedId(sessionData[0].id);
      }
    } catch (err) {
      setError(err.message);
    }
  }

  async function generateDemo() {
    try {
      await postJson("/simulate/session", { num_users: 12, sessions_per_user: 2, seed: 7, persist: true });
      await postJson("/train/baseline", { seed: 7, num_users: 40, sessions_per_user: 3 });
      await postJson("/evaluate", { seed: 11, num_users: 24, sessions_per_user: 2 });
      await refresh();
    } catch (err) {
      setError(err.message);
    }
  }

  const filteredSessions = sessions.filter((session) => {
    if (filters.label && session.label !== filters.label) return false;
    return true;
  });

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <p className="eyebrow">Local Research Prototype</p>
          <h1>DistillShield</h1>
          <p className="lede">Adaptive output protection against LLM distillation.</p>
        </div>
        <button className="primary" onClick={generateDemo}>Generate Demo Data</button>
        <button className="secondary" onClick={refresh}>Refresh</button>
        <div className="filters">
          <label>
            Class
            <select value={filters.label} onChange={(event) => setFilters({ ...filters, label: event.target.value })}>
              <option value="">All</option>
              <option value="normal">normal</option>
              <option value="laboratory_legitimate">laboratory_legitimate</option>
              <option value="suspicious">suspicious</option>
              <option value="high_threat">high_threat</option>
            </select>
          </label>
        </div>
        <div className="session-list">
          {filteredSessions.map((session) => (
            <button
              key={session.id}
              className={`session-card ${selectedId === session.id ? "selected" : ""}`}
              onClick={() => setSelectedId(session.id)}
            >
              <strong>{session.label || "unknown"}</strong>
              <span>{session.user_id}</span>
              <span>{new Date(session.started_at).toLocaleString()}</span>
            </button>
          ))}
        </div>
      </aside>
      <main className="main-panel">
        {error ? <div className="error">{error}</div> : null}
        <section className="top-grid">
          <Card title="Latest Experiment">
            {experiment ? (
              <div>
                <p><strong>{experiment.experiment_id}</strong></p>
                <Metric label="Accuracy" value={experiment.metrics.accuracy} />
                <Metric label="F1" value={experiment.metrics.f1} />
                <Metric label="Leakage" value={experiment.metrics.leakage_proxy_mean} />
              </div>
            ) : <p>No experiment run yet.</p>}
          </Card>
          <Card title="Models">
            {models.length ? models.map((model) => (
              <div key={model.name} className="metric-row">
                <span>{model.name}</span>
                <span>{model.metrics.f1?.toFixed?.(3) ?? "n/a"}</span>
              </div>
            )) : <p>No trained models.</p>}
          </Card>
        </section>

        <section className="detail-grid">
          <Card title="Session Detail">
            {detail ? <SessionDetail detail={detail} /> : <p>Select a session.</p>}
          </Card>
          <Card title="Features / Risk / Transformation">
            {detail ? <AssessmentDetail detail={detail} /> : <p>Score a session through the API to populate this view.</p>}
          </Card>
        </section>
      </main>
    </div>
  );
}

function Card({ title, children }) {
  return (
    <div className="card">
      <h2>{title}</h2>
      {children}
    </div>
  );
}

function Metric({ label, value }) {
  return (
    <div className="metric-row">
      <span>{label}</span>
      <strong>{typeof value === "number" ? value.toFixed(3) : value ?? "n/a"}</strong>
    </div>
  );
}

function SessionDetail({ detail }) {
  return (
    <div className="stack">
      <p><strong>Session:</strong> {detail.session.id}</p>
      <p><strong>User:</strong> {detail.session.user_id}</p>
      <p><strong>Label:</strong> {detail.session.label || "unknown"}</p>
      <p><strong>Org:</strong> {detail.api_context?.org_id}</p>
      <p><strong>Source:</strong> {detail.api_context?.source}</p>
      <h3>Queries</h3>
      {detail.queries?.map((query) => (
        <div key={query.id} className="query-block">
          <p>{query.text}</p>
          <small>{new Date(query.timestamp).toLocaleString()}</small>
        </div>
      ))}
    </div>
  );
}

function AssessmentDetail({ detail }) {
  const featureItems = Object.values(detail.features?.feature_payload || {});
  return (
    <div className="stack">
      <p><strong>Risk score:</strong> {detail.risk_assessment?.risk_score ?? "n/a"}</p>
      <p><strong>Predicted class:</strong> {detail.risk_assessment?.predicted_class ?? "n/a"}</p>
      <p><strong>Confidence:</strong> {detail.risk_assessment?.confidence ?? "n/a"}</p>
      <p><strong>Policy:</strong> {detail.policy_decision?.chosen_policy ?? "n/a"}</p>
      <p><strong>Leakage proxy:</strong> {detail.transformation?.leakage_proxy_score ?? "n/a"}</p>
      <h3>Reasons</h3>
      {(detail.risk_assessment?.reasons || []).map((reason, index) => (
        <div key={index} className="metric-row">
          <span>{reason.reason}</span>
          <strong>{reason.contribution}</strong>
        </div>
      ))}
      <h3>Feature Values</h3>
      <div className="feature-list">
        {featureItems.map((feature) => (
          <div key={feature.name} className="feature-row">
            <span>{feature.name}</span>
            <span>{Number(feature.value).toFixed(3)}</span>
            <small>{feature.provenance}</small>
          </div>
        ))}
      </div>
      <h3>Outputs</h3>
      <div className="output-grid">
        <div>
          <h4>Raw teacher output</h4>
          <pre>{detail.transformation?.raw_output || "Not generated yet."}</pre>
        </div>
        <div>
          <h4>Transformed output</h4>
          <pre>{detail.transformation?.transformed_output || "Not generated yet."}</pre>
        </div>
      </div>
    </div>
  );
}
