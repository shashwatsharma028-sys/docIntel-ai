import { useState, useCallback } from "react"
import { useDropzone } from "react-dropzone"
import { FileText, Upload, CheckCircle, Loader } from "lucide-react"

const MODES = [
  { id: "enterprise", label: "🏢 Enterprise / SOP", desc: "Company policies & manuals" },
  { id: "legal", label: "⚖️ Legal / Contracts", desc: "Simplify legal language" },
  { id: "medical", label: "🏥 Medical Reports", desc: "Explain medical documents" },
  { id: "study", label: "🎓 Study Assistant", desc: "Learn from study material" },
]

export default function Sidebar({ uploadedFiles, onUpload, mode, onModeChange }) {
  const [uploading, setUploading] = useState(false)
  const [status, setStatus] = useState("")

  const onDrop = useCallback(async (acceptedFiles) => {
    const file = acceptedFiles[0]
    if (!file) return
    setUploading(true)
    setStatus("Uploading & indexing...")
    const result = await onUpload(file)
    setStatus(result.success ? "Indexed successfully!" : result.message)
    setUploading(false)
    setTimeout(() => setStatus(""), 3000)
  }, [onUpload])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { "application/pdf": [".pdf"] },
    maxFiles: 1,
  })

  return (
    <div className="w-72 bg-gray-900 border-r border-gray-800 flex flex-col p-4 gap-4">
      <div className="flex items-center gap-2 pb-3 border-b border-gray-800">
        <FileText className="text-blue-400" size={20} />
        <h1 className="font-semibold text-lg">DocIntel AI</h1>
      </div>

      {/* Mode Selector */}
      <div>
        <p className="text-xs text-gray-500 uppercase tracking-wider mb-2">Document Mode</p>
        <div className="space-y-1">
          {MODES.map((m) => (
            <button
              key={m.id}
              onClick={() => onModeChange(m.id)}
              className={`w-full text-left px-3 py-2 rounded-lg transition-colors text-sm
                ${mode === m.id ? "bg-blue-600 text-white" : "hover:bg-gray-800 text-gray-300"}`}
            >
              <div className="font-medium">{m.label}</div>
              <div className={`text-xs ${mode === m.id ? "text-blue-200" : "text-gray-500"}`}>{m.desc}</div>
            </button>
          ))}
        </div>
      </div>

      {/* Upload */}
      <div
        {...getRootProps()}
        className={`border-2 border-dashed rounded-xl p-4 text-center cursor-pointer transition-colors
          ${isDragActive ? "border-blue-400 bg-blue-950" : "border-gray-700 hover:border-gray-500"}`}
      >
        <input {...getInputProps()} />
        <Upload className="mx-auto mb-2 text-gray-400" size={20} />
        <p className="text-xs text-gray-400">
          {isDragActive ? "Drop PDF here" : "Drag & drop a PDF or click to upload"}
        </p>
      </div>

      {uploading && (
        <div className="flex items-center gap-2 text-sm text-blue-400">
          <Loader size={14} className="animate-spin" />{status}
        </div>
      )}
      {!uploading && status && (
        <div className="flex items-center gap-2 text-sm text-green-400">
          <CheckCircle size={14} />{status}
        </div>
      )}

      {uploadedFiles.length > 0 && (
        <div className="flex-1 overflow-y-auto">
          <p className="text-xs text-gray-500 uppercase tracking-wider mb-2">Indexed Documents</p>
          {uploadedFiles.map((f, i) => (
            <div key={i} className="flex items-center gap-2 py-2 px-3 rounded-lg bg-gray-800 mb-1">
              <FileText size={14} className="text-blue-400 flex-shrink-0" />
              <span className="text-sm text-gray-300 truncate">{f}</span>
            </div>
          ))}
        </div>
      )}

      <div className="mt-auto pt-3 border-t border-gray-800">
        <p className="text-xs text-gray-600">Gemini 2.0 Flash · Groq fallback</p>
      </div>
    </div>
  )
}