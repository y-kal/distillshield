const API_BASE = "http://localhost:8000";

async function parseError(response, path) {
  const body = await response.text();
  const detail = body ? `: ${body}` : "";
  return new Error(`API ${response.status} for ${path}${detail}`);
}

export async function fetchJson(path) {
  let response;
  try {
    response = await fetch(`${API_BASE}${path}`);
  } catch (error) {
    throw new Error(`Unable to reach API at ${API_BASE}${path}: ${error.message}`);
  }
  if (!response.ok) {
    throw await parseError(response, path);
  }
  return response.json();
}

export async function postJson(path, body) {
  let response;
  try {
    response = await fetch(`${API_BASE}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
  } catch (error) {
    throw new Error(`Unable to reach API at ${API_BASE}${path}: ${error.message}`);
  }
  if (!response.ok) {
    throw await parseError(response, path);
  }
  return response.json();
}
