import { useState } from 'react'

function Chatbot() {
    const [question, setQuestion] = useState('')
    const [messages, setMessages] = useState([])
    const [loading, setLoading] = useState(false)

    const handleAsk = async () => {
        if (!question.trim() || loading) return

        const currentQuestion = question
        setQuestion('')
        setLoading(true)

        try {
            const response = await fetch('https://d3n25zqy4yie6v.cloudfront.net/chat/ask', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ question: currentQuestion})
            })
            const data = await response.json()
            setMessages((prev) => [...prev, { question: currentQuestion, answer: data.answer }])
        } catch (error) {
            setMessages((prev) => [
                ...prev,
                { question: currentQuestion, answer: 'Error: could not reach the server.' }
            ])
        } finally {
            setLoading(false)
        }
    }

    const handleKeyDown = (e) => {
        if (e.key === 'Enter') handleAsk()
    }

    return (
        <div className='chatbot-section'>
            <h2>Medical Knowledge Assistant</h2>
            <p>Ask a question about heart disease, symptoms, or treatment.</p>

            <div className='chat-window'>
                {messages.length === 0 && !loading && (
                    <p className='caht-placeholder'>No Messages yet - Ask a question to get started!</p>
                )}

                {messages.map((msg, index) => (
                    <div key={index} className='chat-exchange'>
                        <p className='chat-question'><strong>You:</strong>{msg.question}</p>
                        <p className='chat-answer'><strong>Assistant:</strong>{msg.answer}</p>
                    </div>
                ))}

                {loading && <p className='chat-loading'>Thinking...</p>}
            </div>

            <div className='chat-input-row'>
                <input
                    type="text"
                    value={question}
                    onChange={(e) => setQuestion(e.target.value)}
                    onKeyDown={handleKeyDown}
                    placeholder='e.g. What are the symptoms of angina?'
                    disabled={loading}
                    />
                    <button type="button" onClick={handleAsk} disabled={loading}>
                        {loading ? 'Asking...' : 'Ask'}
                    </button>
            </div>
        </div>
    )
}

export default Chatbot