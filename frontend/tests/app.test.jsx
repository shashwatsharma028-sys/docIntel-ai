// Baseline workflow tests for the React UI. fetch is mocked: no backend or network is used.
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest"
import { render, screen, cleanup, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import App from "../src/App.jsx"

const API = "http://127.0.0.1:8000"
const jsonResponse = (body, init = {}) => Promise.resolve({ ok: init.ok ?? true, status: init.status ?? 200, json: () => Promise.resolve(body) })

let fetchMock
beforeEach(() => {
  fetchMock = vi.fn()
  globalThis.fetch = fetchMock
  Element.prototype.scrollIntoView = vi.fn()
})
afterEach(cleanup)

const pdf = (name = "policy.pdf") => new File(["%PDF-1.4"], name, { type: "application/pdf" })
const fileInput = (container) => container.querySelector('input[type="file"]')
const textbox = () => screen.getByPlaceholderText("Ask anything about your documents...")

describe("initial render", () => {
  it("shows the four modes with enterprise selected and the empty-chat prompt", () => {
    render(<App />)
    for (const label of ["🏢 Enterprise / SOP", "⚖️ Legal / Contracts", "🏥 Medical Reports", "🎓 Study Assistant"]) {
      expect(screen.getByText(label)).toBeTruthy()
    }
    expect(screen.getByText("🏢 Enterprise / SOP").closest("button").className).toContain("bg-blue-600")
    expect(screen.getByText("Upload a PDF and start asking questions")).toBeTruthy()
    expect(screen.queryByText("Indexed Documents")).toBeNull()
  })
})

describe("chat workflow", () => {
  it("POSTs question and current mode to /chat and renders answer, confidence, sources, model", async () => {
    const user = userEvent.setup()
    fetchMock.mockReturnValue(jsonResponse({ answer: "Five days carry forward.", model: "gemini-2.0-flash", confidence: "high", sources: ["policy.pdf", "hr.pdf"] }))
    render(<App />)
    await user.click(screen.getByText("⚖️ Legal / Contracts"))
    await user.type(textbox(), "How much leave?{Enter}")

    expect(fetchMock).toHaveBeenCalledWith(`${API}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: "How much leave?", mode: "legal" }),
    })
    expect(await screen.findByText("Five days carry forward.")).toBeTruthy()
    expect(screen.getByText("How much leave?")).toBeTruthy()
    const conf = screen.getByText("● high confidence")
    expect(conf.className).toContain("text-green-400")
    expect(screen.getByText("2 sources")).toBeTruthy()
    expect(screen.getByText("· gemini-2.0-flash")).toBeTruthy()
    expect(screen.queryByText("hr.pdf")).toBeNull()
    await user.click(screen.getByText("2 sources"))
    expect(screen.getByText("policy.pdf")).toBeTruthy()
    expect(screen.getByText("hr.pdf")).toBeTruthy()
    expect(textbox().value).toBe("")
  })

  it("uses singular 'source' and maps medium/low confidence colours", async () => {
    const user = userEvent.setup()
    fetchMock
      .mockReturnValueOnce(jsonResponse({ answer: "A1", model: "m", confidence: "medium", sources: ["one.pdf"] }))
      .mockReturnValueOnce(jsonResponse({ answer: "A2", model: "m", confidence: "low", sources: [] }))
    render(<App />)
    await user.type(textbox(), "q1{Enter}")
    await screen.findByText("A1")
    expect(screen.getByText("1 source")).toBeTruthy()
    expect(screen.getByText("● medium confidence").className).toContain("text-yellow-400")
    await user.type(textbox(), "q2{Enter}")
    await screen.findByText("A2")
    expect(screen.getByText("● low confidence").className).toContain("text-red-400")
  })

  it("shows the backend-unreachable message when fetch rejects", async () => {
    const user = userEvent.setup()
    fetchMock.mockRejectedValue(new TypeError("Failed to fetch"))
    render(<App />)
    await user.type(textbox(), "hello{Enter}")
    expect(await screen.findByText("Something went wrong. Is the backend running?")).toBeTruthy()
  })

  it("shows the 'Thinking...' indicator while waiting and blocks double-send", async () => {
    const user = userEvent.setup()
    let resolve
    fetchMock.mockReturnValue(new Promise((r) => { resolve = r }))
    render(<App />)
    await user.type(textbox(), "first{Enter}")
    expect(screen.getByText("Thinking...")).toBeTruthy()
    await user.type(textbox(), "second{Enter}")
    expect(fetchMock).toHaveBeenCalledTimes(1)
    resolve({ ok: true, json: () => Promise.resolve({ answer: "done", model: "m", confidence: "low", sources: [] }) })
    await screen.findByText("done")
    expect(screen.queryByText("Thinking...")).toBeNull()
  })

  it("ignores blank input; Shift+Enter inserts a newline instead of sending", async () => {
    const user = userEvent.setup()
    render(<App />)
    await user.type(textbox(), "   {Enter}")
    await user.type(textbox(), "line1{Shift>}{Enter}{/Shift}line2")
    expect(fetchMock).not.toHaveBeenCalled()
    expect(textbox().value).toContain("line1\n")
  })

  it("switching mode clears the conversation", async () => {
    const user = userEvent.setup()
    fetchMock.mockReturnValue(jsonResponse({ answer: "ans", model: "m", confidence: "low", sources: [] }))
    render(<App />)
    await user.type(textbox(), "q{Enter}")
    await screen.findByText("ans")
    await user.click(screen.getByText("🏥 Medical Reports"))
    expect(screen.queryByText("ans")).toBeNull()
    expect(screen.getByText("Upload a PDF and start asking questions")).toBeTruthy()
    expect(screen.getByText("🏥 Medical Reports").closest("button").className).toContain("bg-blue-600")
  })
})

describe("upload workflow", () => {
  it("POSTs multipart 'file' to /upload, lists the document and shows success", async () => {
    const user = userEvent.setup()
    fetchMock.mockReturnValue(jsonResponse({ message: "policy.pdf uploaded and indexed successfully" }))
    const { container } = render(<App />)
    await user.upload(fileInput(container), pdf())
    await screen.findByText("Indexed successfully!")
    const [url, opts] = fetchMock.mock.calls[0]
    expect(url).toBe(`${API}/upload`)
    expect(opts.method).toBe("POST")
    expect(opts.body.get("file").name).toBe("policy.pdf")
    expect(screen.getByText("Indexed Documents")).toBeTruthy()
    expect(screen.getByText("policy.pdf")).toBeTruthy()
  })

  it("network failure shows 'Upload failed' and does not list the file", async () => {
    const user = userEvent.setup()
    fetchMock.mockRejectedValue(new TypeError("Failed to fetch"))
    const { container } = render(<App />)
    await user.upload(fileInput(container), pdf("x.pdf"))
    expect(await screen.findByText("Upload failed")).toBeTruthy()
    expect(screen.queryByText("x.pdf")).toBeNull()
  })

  it("dropzone accepts only PDFs", async () => {
    const user = userEvent.setup()
    const { container } = render(<App />)
    expect(fileInput(container).getAttribute("accept")).toContain("application/pdf")
    await user.upload(fileInput(container), new File(["x"], "notes.txt", { type: "text/plain" }))
    expect(fetchMock).not.toHaveBeenCalled()
  })

  it("does not change mode or clear chat on upload", async () => {
    const user = userEvent.setup()
    fetchMock
      .mockReturnValueOnce(jsonResponse({ answer: "kept", model: "m", confidence: "low", sources: [] }))
      .mockReturnValueOnce(jsonResponse({ message: "ok" }))
    const { container } = render(<App />)
    await user.type(textbox(), "q{Enter}")
    await screen.findByText("kept")
    await user.upload(fileInput(container), pdf())
    await screen.findByText("Indexed successfully!")
    expect(screen.getByText("kept")).toBeTruthy()
  })
})

// Known frontend defects: `it.fails` passes ONLY while the defect exists (strict expected-failure).
describe("known defects", () => {
  it.fails("D-13: an HTTP error response from /upload is not reported as success", async () => {
    const user = userEvent.setup()
    fetchMock.mockReturnValue(jsonResponse({ detail: "boom" }, { ok: false, status: 422 }))
    const { container } = render(<App />)
    await user.upload(fileInput(container), pdf("bad.pdf"))
    await waitFor(() => expect(fetchMock).toHaveBeenCalled())
    await new Promise((r) => setTimeout(r, 50))
    expect(screen.queryByText("Indexed successfully!")).toBeNull()
    expect(screen.queryByText("bad.pdf")).toBeNull()
  })

  it.fails("D-14: an upload failure message is not rendered with success (green) styling", async () => {
    const user = userEvent.setup()
    fetchMock.mockRejectedValue(new TypeError("Failed to fetch"))
    const { container } = render(<App />)
    await user.upload(fileInput(container), pdf())
    const msg = await screen.findByText("Upload failed")
    expect(msg.className).not.toContain("text-green-400")
  })
})
