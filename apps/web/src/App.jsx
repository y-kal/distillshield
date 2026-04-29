import { useEffect, useState } from "react";
import { fetchJson, postJson } from "./api";

const emptyFilters = { label: "" };

const CLASS_TONE = {
  normal: "safe",
  laboratory_legitimate: "controlled",
  suspicious: "elevated",
  high_threat: "severe",
};

export default function App() {
  const [sessions, setSessions] = useState([]);
  const [selectedId, setSelectedId] = useState(null);
  const [detail, setDetail] = useState(null);
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
      const [sessionData, latestExperiment] = await Promise.all([
        fetchJson("/sessions"),
        fetchJson("/experiments/latest"),
      ]);
      setSessions(sessionData);
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
          <p className="lede">Rule-based adaptive output protection for reasoning leakage risk.</p>
        </div>
        <button className="primary" onClick={generateDemo}>Generate Demo Scenarios</button>
        <button className="secondary" onClick={refresh}>Refresh</button>
        <div className="filters">
          <label>
            Behaviour class
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
              <span>{CLASS_TONE[session.label] || "unclassified"}</span>
              <span>{session.user_id}</span>
              <span>{new Date(session.started_at).toLocaleString()}</span>
            </button>
          ))}
        </div>
      </aside>
      <main className="main-panel">
        {error ? <div className="error">{error}</div> : null}
        <section className="top-grid">
          <Card title="Latest Evaluation">
            {experiment ? (
              <div>
                <p><strong>{experiment.experiment_id}</strong></p>
                <Metric label="Scenario count" value={experiment.metrics.scenario_count} />
                <Metric label="Mean risk score" value={experiment.metrics.mean_risk_score} />
                <Metric label="Leakage reduction" value={experiment.metrics.leakage_proxy_reduction_mean} />
                <Metric label="Utility preservation" value={experiment.metrics.utility_preservation_mean} />
              </div>
            ) : <p>No evaluation run yet.</p>}
          </Card>
          <Card title="Rule Engine">
            <p><strong>Model:</strong> Grouped rule-based risk engine</p>
            <p><strong>Decision flow:</strong> category scores, escalation rules, policy engine, protected output</p>
            <p><strong>Confidence:</strong> derived from rule agreement, not ML probabilities</p>
          </Card>
        </section>

        <section className="detail-grid">
          <Card title="Session Detail">
            {detail ? <SessionDetail detail={detail} /> : <p>Select a session.</p>}
          </Card>
          <Card title="Assessment Detail">
            {detail ? <AssessmentDetail detail={detail} /> : <p>Select a session to inspect its assessment.</p>}
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
      <p><strong>Scenario label:</strong> {detail.session.label || "unknown"}</p>
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
  const risk = detail.risk_assessment;
  const policy = detail.policy_decision;
  const transformation = detail.transformation;
  const categoryScores = risk?.category_scores || {};

  return (
    <div className="stack">
      <div className={`status-banner ${CLASS_TONE[risk?.predicted_class] || "safe"}`}>
        <strong>{risk?.predicted_class || "unclassified"}</strong>
        <span>{CLASS_TONE[risk?.predicted_class] || "status unavailable"}</span>
      </div>
      <p><strong>Risk score:</strong> {risk?.risk_score ?? "n/a"}</p>
      <p><strong>Risk confidence:</strong> {risk?.confidence ?? "n/a"}</p>
      <p><strong>Policy decision:</strong> {policy?.chosen_policy ?? "n/a"}</p>
      <p><strong>Policy reason:</strong> {policy?.policy_reason ?? "n/a"}</p>
      <p><strong>Leakage proxy:</strong> {transformation?.leakage_proxy_score ?? "n/a"}</p>

      <h3>Category Scores</h3>
      {["query_pattern", "reasoning_extraction", "automation", "infrastructure", "legitimate_use"].map((name) => (
        <div key={name} className="metric-row">
          <span>{name}</span>
          <strong>{typeof categoryScores[name] === "number" ? categoryScores[name].toFixed(3) : "n/a"}</strong>
        </div>
      ))}

      <h3>Top Reasons</h3>
      {(risk?.top_reasons || []).length ? (risk.top_reasons.map((reason, index) => (
        <div key={index} className="query-block compact">
          {reason}
        </div>
      ))) : <p>No top reasons available.</p>}

      <h3>Triggered Rules</h3>
      {(risk?.triggered_rules || []).length ? (risk.triggered_rules.map((rule) => (
        <div key={rule.id} className="query-block compact">
          <strong>{rule.id}</strong>
          <p>{rule.description}</p>
          <small>{rule.effect}</small>
        </div>
      ))) : <p>No escalation rules triggered.</p>}

      <h3>Risk Reducers</h3>
      {(risk?.risk_reducers || []).length ? (risk.risk_reducers.map((item, index) => (
        <div key={index} className="query-block compact">
          {item}
        </div>
      ))) : <p>No explicit risk reducers recorded.</p>}

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
          <h4>Mock teacher output</h4>
          <pre>{transformation?.raw_output || "Not generated yet."}</pre>
        </div>
        <div>
          <h4>Protected output</h4>
          <pre>{transformation?.transformed_output || "Not generated yet."}</pre>
        </div>
      </div>
    </div>
  );
}
