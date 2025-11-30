import React, { useEffect, useRef } from 'react';

const styles = {
    container: {
        backgroundColor: '#0c0c0c',
        color: '#cccccc',
        fontFamily: 'Consolas, Monaco, "Andale Mono", "Ubuntu Mono", monospace',
        fontSize: '0.9rem',
        padding: '1rem',
        borderRadius: '6px',
        border: '1px solid #333',
        height: '400px',
        overflowY: 'auto',
        whiteSpace: 'pre-wrap', // Permet le retour à la ligne
        display: 'flex',
        flexDirection: 'column',
        gap: '2px'
    },
    line: {
        lineHeight: '1.4',
        wordBreak: 'break-all'
    },
    empty: {
        color: '#555',
        fontStyle: 'italic',
        textAlign: 'center',
        marginTop: '2rem'
    }
};

const Terminal = ({ logs }) => {
    const bottomRef = useRef(null);

    // Auto-scroll vers le bas quand les logs changent
    useEffect(() => {
        if (bottomRef.current) {
            bottomRef.current.scrollIntoView({ behavior: 'smooth' });
        }
    }, [logs]);

    return (
        <div style={styles.container}>
            {logs.length === 0 && (
                <div style={styles.empty}>
                    Ready to start. Logs will appear here...
                </div>
            )}
            
            {logs.map((line, index) => (
                <div key={index} style={styles.line}>
                    {line}
                </div>
            ))}
            
            {/* Élément invisible pour scroller jusqu'ici */}
            <div ref={bottomRef} />
        </div>
    );
};

export default Terminal;