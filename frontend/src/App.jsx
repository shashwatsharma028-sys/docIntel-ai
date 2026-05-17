import { useState } from "react"
import Sidebar from "./components/Sidebar"
import Chat from "./components/Chat"

export default function App() {
  const [uploadedFiles, setUploadedFiles] = useState([])
  const [messages, setMessages] = useState([])
  const [loading, setLoading] = useState(false)
  const [mode, setMode] = useState("enterprise")

  const handleUpload = async (file) => {
    const formData = new FormData()
    formData.append("file", file)
    try {
      const res = await fetch("http://127.0.0.1:8000/upload", {
        method: "POST",
        body: formData,
      })
      const data = await res.json()
      setUploadedFiles((prev) => [...prev, file.name])
      return { success: true, message: data.message }
    } catch (e) {
      return { success: false, message: "Upload failed" }
    }
  }

  const handleChat = async (question) => {
    setMessages((prev) => [...prev, { role: "user", text: question }])
    setLoading(true)
    try {
      const res = await fetch("http://127.0.0.1:8000/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, mode }),
      })
      const data = await res.json()
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          text: data.answer,
          sources: data.sources,
          model: data.model,
          confidence: data.confidence
        },
      ])
    } catch (e) {
      setMessages((prev) => [
        ...prev,
        { role: "assistant", text: "Something went wrong. Is the backend running?", sources: [] },
      ])
    }
    setLoading(false)
  }

  const handleModeChange = (newMode) => {
    setMode(newMode)
    setMessages([])
  }

  return (
    <div className="flex h-screen bg-gray-950 text-gray-100">
      <Sidebar
        uploadedFiles={uploadedFiles}
        onUpload={handleUpload}
        mode={mode}
        onModeChange={handleModeChange}
      />
      <Chat messages={messages} onSend={handleChat} loading={loading} mode={mode} />
    </div>
  )
}