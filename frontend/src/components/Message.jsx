import { useState } from "react"
import { User, Bot, ChevronDown, ChevronUp, FileText } from "lucide-react"

const CONFIDENCE_COLORS = {
  high: "text-green-400",
  medium: "text-yellow-400",
  low: "text-red-400"
}

export default function Message({ message }) {
  const [showSources, setShowSources] = useState(false)
  const isUser = message.role === "user"

  return (
    <div className={`flex gap-3 ${isUser ? "flex-row-reverse" : ""}`}>
      <div className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0
        ${isUser ? "bg-blue-600" : "bg-gray-700"}`}>
        {isUser ? <User size={14} /> : <Bot size={14} />}
      </div>

      <div className={`max-w-2xl space-y-2 ${isUser ? "items-end" : "items-start"} flex flex-col`}>
        <div className={`px-4 py-3 rounded-2xl text-sm leading-relaxed whitespace-pre-wrap
          ${isUser ? "bg-blue-600 text-white rounded-tr-sm" : "bg-gray-800 text-gray-100 rounded-tl-sm"}`}>
          {message.text}
        </div>

        {!isUser && (
          <div className="flex items-center gap-3 flex-wrap">
            {message.confidence && (
              <span className={`text-xs ${CONFIDENCE_COLORS[message.confidence]}`}>
                ● {message.confidence} confidence
              </span>
            )}
            {message.sources?.length > 0 && (
              <button
                onClick={() => setShowSources(!showSources)}
                className="flex items-center gap-1 text-xs text-gray-500 hover:text-gray-400 transition-colors"
              >
                {showSources ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
                {message.sources.length} source{message.sources.length > 1 ? "s" : ""}
                {message.model && <span className="ml-2 text-gray-600">· {message.model}</span>}
              </button>
            )}
          </div>
        )}

        {showSources && (
          <div className="space-y-1 w-full">
            {message.sources.map((s, i) => (
              <div key={i} className="flex items-center gap-2 px-3 py-1.5 bg-gray-800 rounded-lg text-xs text-gray-400">
                <FileText size={12} className="text-blue-400" />
                {s}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}