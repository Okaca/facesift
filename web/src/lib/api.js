const BASE = import.meta.env.BASE_URL;

export async function fetchPhotos(offset, limit = 50) {
  const params = new URLSearchParams({ offset, limit });
  const res = await fetch(`${BASE}api/photos?${params}`);

  if (!res.ok) {
    throw new Error(`Could not load photos (${res.status})`);
  }

  return res.json();
}

export async function searchSelfie(file) {
  const form = new FormData();
  form.append("file", file); // "file" must match `file: UploadFile` in FastAPI

  // no Content-Type header: the browser sets multipart/form-data with its boundary
  const res = await fetch(`${BASE}api/search`, { method: "POST", body: form });

  if (!res.ok) {
    // our own errors have a string detail ("No face found, ..."); FastAPI's
    // validation errors have a list, and a proxy error page may not be JSON at all
    const body = await res.json().catch(() => null);
    const detail =
      typeof body?.detail === "string"
        ? body.detail
        : `Search failed (${res.status})`;
    throw new Error(detail);
  }

  return res.json();
}
