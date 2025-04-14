import { useState, useEffect } from 'react'
import axios from 'axios'
import toast, { Toaster } from 'react-hot-toast'

function App() {
  const [query, setQuery] = useState('')
  const [results, setResults] = useState([])
  const [topK, setTopK] = useState(5)
  const [rerank, setRerank] = useState(true)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [darkMode, setDarkMode] = useState(() => {
    return JSON.parse(localStorage.getItem('darkMode')) || false
  })

  const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

  useEffect(() => {
    localStorage.setItem('darkMode', JSON.stringify(darkMode))
    document.documentElement.classList.toggle('dark', darkMode)
  }, [darkMode])

  const Spinner = () => (
    <svg className="animate-spin h-5 w-5 text-white" viewBox="0 0 24 24">
      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.4 0 0 5.4 0 12h4z" />
    </svg>
  )

  const handleSearch = async (e) => {
    e.preventDefault()
    setLoading(true)
    setError('')
    try {
      const response = await axios.post(`${API_URL}/search`, {
        query,
        top_k: topK,
        rerank
      }, {
        headers: {
          'X-API-Key': import.meta.env.VITE_GOOGLE_API_KEY
        }
      })
      setResults(response.data)
    } catch (err) {
      setError(err.response?.data?.detail || 'Search failed')
    } finally {
      setLoading(false)
    }
  }

  const sendFeedback = async (sentenceHash, isRelevant) => {
    try {
      await axios.post(`${API_URL}/feedback`, {
        query,
        sentence_hash: sentenceHash,
        is_relevant: isRelevant
      })
      toast.success(isRelevant ? 'Marked Relevant' : 'Marked Not Relevant')
    } catch (err) {
      toast.error('Feedback failed')
    }
  }

  return (
    <div className={`min-h-screen transition-colors ${darkMode ? 'bg-gray-900 text-white' : 'bg-white text-gray-900'}`}>
      <Toaster position="top-right" />
      <div className="container mx-auto p-6">
        <header className="flex justify-between items-center mb-6">
          <h1 className="text-3xl font-bold">Resume Search</h1>
          <button
            onClick={() => setDarkMode(!darkMode)}
            className="px-4 py-2 border rounded hover:bg-gray-200 dark:hover:bg-gray-700"
          >
            {darkMode ? 'Light Mode' : 'Dark Mode'}
          </button>
        </header>

        <form onSubmit={handleSearch} className="space-y-4 mb-6">
          <div className="flex gap-3">
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Enter job requirement..."
              className="flex-1 p-3 border rounded"
            />
            <button
              type="submit"
              disabled={loading}
              className="bg-blue-600 text-white px-5 py-3 rounded flex items-center gap-2"
            >
              {loading ? <Spinner /> : 'Search'}
            </button>
          </div>
          <div className="flex gap-6 items-center">
            <label className="flex items-center gap-2">
              Results:
              <input
                type="number"
                min="1"
                value={topK}
                onChange={(e) => setTopK(Math.max(1, Number(e.target.value)))}
                className="w-16 p-2 border rounded"
              />
            </label>
            <label className="flex items-center gap-2">
              <input
                type="checkbox"
                checked={rerank}
                onChange={(e) => setRerank(e.target.checked)}
              />
              Use LLM Reranking
            </label>
          </div>
        </form>

        {error && <div className="text-red-500 bg-red-100 p-3 rounded mb-4">{error}</div>}

        {loading && (
          <div className="space-y-4">
            {[...Array(3)].map((_, i) => (
              <div key={i} className="animate-pulse h-24 bg-gray-200 dark:bg-gray-700 rounded"></div>
            ))}
          </div>
        )}

        {!loading && results.length > 0 && (
          <div className="space-y-6">
            {results.map((r) => (
              <div key={r.sentence_hash} className="p-5 border rounded-lg shadow-md dark:border-gray-700 dark:bg-gray-800">
                <div className="flex justify-between items-center mb-2">
                  <div>
                    <p className="text-sm text-gray-400">Source: {r.source || 'Unknown'}</p>
                    <p className="text-sm text-gray-500">Score: {r.score?.toFixed(4)}</p>
                  </div>
                  <div className="flex gap-2">
                    <button
                      onClick={() => sendFeedback(r.sentence_hash, true)}
                      className="bg-green-500 text-white px-3 py-1 rounded text-sm hover:bg-green-600"
                    >
                      Relevant
                    </button>
                    <button
                      onClick={() => sendFeedback(r.sentence_hash, false)}
                      className="bg-red-500 text-white px-3 py-1 rounded text-sm hover:bg-red-600"
                    >
                      Not Relevant
                    </button>
                  </div>
                </div>
                <p className="mb-2">{r.text}</p>
                {r.justification && (
                  <div className="bg-gray-100 dark:bg-gray-700 p-3 rounded">
                    <p className="text-sm text-gray-600 dark:text-gray-300">{r.justification}</p>
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

export default App
