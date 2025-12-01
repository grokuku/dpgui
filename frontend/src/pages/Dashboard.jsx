import React, { useEffect, useState, useRef } from 'react';
import apiClient from '../api';
import {
    Activity, Cpu, HardDrive, Play, Square, Trash2, FileText, X,
    Thermometer, Fan, Zap, Clock
} from 'lucide-react';
import Terminal from '../components/Terminal';

// --- UI Components ---
const ProgressBar = ({ percent, color = '#646cff', label, subLabel }) => (
    <div style={{ marginBottom: '0.8rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.2rem', fontSize: '0.8rem', color: '#ccc' }}>
            <span>{label}</span>
            <span style={{ textAlign: 'right' }}>
                {subLabel && <span style={{ color: '#777', marginRight: '8px' }}>{subLabel}</span>}
                <span>{percent}%</span>
            </span>
        </div>
        <div style={{ height: '6px', background: '#333', borderRadius: '3px', overflow: 'hidden' }}>
            <div style={{ width: `${Math.min(100, Math.max(0, percent))}%`, height: '100%', background: color, transition: 'width 0.5s ease' }} />
        </div>
    </div>
);

const StatusBadge = ({ status }) => {
    const colors = { pending: '#f1c40f', running: '#3498db', completed: '#2ecc71', failed: '#e74c3c', stopped: '#95a5a6' };
    return <span style={{ backgroundColor: colors[status] || '#777', color: '#fff', padding: '2px 8px', borderRadius: '4px', fontSize: '0.7rem', textTransform: 'uppercase', fontWeight: 'bold' }}>{status}</span>;
};

const MetricItem = ({ icon: Icon, value, label, color = "#ccc" }) => (
    <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px' }}>
        <Icon size={16} color={color} />
        <div>
            <div style={{ fontSize: '0.9rem', fontWeight: 'bold', lineHeight: '1' }}>{value}</div>
            <div style={{ fontSize: '0.7rem', color: '#666' }}>{label}</div>
        </div>
    </div>
);

const GpuDetailCard = ({ gpu }) => {
    return (
        <div style={{ background: '#1a1a1a', padding: '1rem', borderRadius: '6px', marginBottom: '1rem', border: '1px solid #333' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem', borderBottom: '1px solid #333', paddingBottom: '0.5rem' }}>
                <strong style={{ color: '#2ecc71' }}>GPU {gpu.id}</strong>
                <span style={{ fontSize: '0.8rem', color: '#aaa' }}>NVIDIA</span>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginBottom: '1rem' }}>
                {/* Left Column: Thermals */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                    <MetricItem icon={Thermometer} label="Temperature" value={`${gpu.temp}°C`} color="#e67e22" />
                    <MetricItem icon={Fan} label="Fan Speed" value={`${gpu.fan}%`} color="#3498db" />
                    <MetricItem icon={Clock} label="Clock Speed" value={`${gpu.clock} MHz`} color="#9b59b6" />
                </div>

                {/* Right Column: Loads */}
                <div>
                    <ProgressBar
                        percent={gpu.usage}
                        label="GPU Load"
                        color="#e74c3c"
                    />
                    <ProgressBar
                        percent={gpu.vram_percent}
                        label="Memory"
                        subLabel={`${(gpu.vram_used / 1024).toFixed(1)} / ${(gpu.vram_total / 1024).toFixed(1)} GB`}
                        color="#3498db"
                    />
                    <MetricItem icon={Zap} label="Power Draw" value={`${gpu.power_draw}W / ${gpu.power_limit}W`} color="#f1c40f" />
                </div>
            </div>
        </div>
    );
};

// --- Log Modal ---
const LogModal = ({ job, onClose }) => {
    const [content, setContent] = useState("Loading logs...");

    useEffect(() => {
        const fetchLogs = async () => {
            try {
                const res = await apiClient.get(`/jobs/${job.id}/logs`);
                setContent(res.data.content);
            } catch (e) {
                setContent("Error fetching logs.");
            }
        };
        fetchLogs();
    }, [job]);

    return (
        <div style={{ position: 'fixed', top: 0, left: 0, right: 0, bottom: 0, background: 'rgba(0,0,0,0.8)', zIndex: 1000, display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
            <div className="card" style={{ width: '80%', height: '80%', display: 'flex', flexDirection: 'column', position: 'relative' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '1rem' }}>
                    <h3>Logs for Job {job.id}</h3>
                    <button onClick={onClose} className="btn-secondary"><X size={20} /></button>
                </div>
                <textarea readOnly value={content} style={{ flex: 1, background: '#111', color: '#0f0', border: 'none', padding: '1rem', fontFamily: 'monospace' }} />
            </div>
        </div>
    );
};

// --- Main Dashboard ---
const Dashboard = () => {
    const [stats, setStats] = useState(null);
    const [jobs, setJobs] = useState([]);
    const [logs, setLogs] = useState([]);
    const [viewLogJob, setViewLogJob] = useState(null);

    const wsRef = useRef(null);
    const activeJobIdRef = useRef(null);

    useEffect(() => {
        const fetchData = async () => {
            try {
                const statsRes = await apiClient.get('/system-stats');
                setStats(statsRes.data);
                const jobsRes = await apiClient.get('/jobs');
                setJobs(jobsRes.data);
            } catch (e) { console.error(e); }
        };
        fetchData();
        const interval = setInterval(fetchData, 2000);
        return () => clearInterval(interval);
    }, []);

    const runningJob = jobs.find(j => j.status === 'running');

    // WebSocket for Running Job Only
    useEffect(() => {
        if (runningJob && activeJobIdRef.current !== runningJob.id) {
            if (wsRef.current) wsRef.current.close();
            activeJobIdRef.current = runningJob.id;
            setLogs([]);
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/api/ws/logs`;
            const ws = new WebSocket(wsUrl);
            ws.onopen = () => setLogs(prev => [...prev, `>>> ATTACHED TO JOB ${runningJob.id} <<<`]);
            ws.onmessage = (event) => { if (event.data !== 'PING') setLogs(prev => [...prev, event.data]); };
            ws.onclose = () => { activeJobIdRef.current = null; };
            wsRef.current = ws;
        } else if (!runningJob && wsRef.current) {
            wsRef.current.close();
            wsRef.current = null;
            activeJobIdRef.current = null;
        }
    }, [runningJob]);

    const handleQueue = async (id) => { await apiClient.post(`/jobs/${id}/queue`); };
    const handleStop = async (id) => { await apiClient.post(`/jobs/${id}/stop`); };
    const handleDelete = async (id) => { if (confirm("Delete job?")) await apiClient.delete(`/jobs/${id}`); };

    // Split Jobs
    const queueJobs = jobs.filter(j => ['pending', 'running'].includes(j.status));
    const poolJobs = jobs.filter(j => !['pending', 'running'].includes(j.status));

    return (
        <div>
            <h2>Dashboard</h2>
            {viewLogJob && <LogModal job={viewLogJob} onClose={() => setViewLogJob(null)} />}

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '1rem', marginBottom: '2rem' }}>
                {/* System & GPU Stats */}
                <div className="card">
                    <h3><Activity size={20} style={{ verticalAlign: 'bottom' }} /> Resources</h3>
                    {stats ? (
                        <>
                            <div style={{ marginBottom: '1.5rem' }}>
                                <ProgressBar percent={stats.cpu} label="System CPU" color="#95a5a6" />
                                <ProgressBar percent={stats.ram.percent} label="System RAM" color="#f1c40f" />
                            </div>

                            {stats.gpu.map((g, idx) => (
                                <GpuDetailCard key={idx} gpu={g} />
                            ))}
                        </>
                    ) : <p>Loading...</p>}
                </div>

                {/* Active Job Terminal */}
                <div className="card" style={{ gridColumn: 'span 2', display: 'flex', flexDirection: 'column' }}>
                    <h3><Cpu size={20} style={{ verticalAlign: 'bottom' }} /> Active Execution</h3>
                    <div style={{ flex: 1 }}>
                        {runningJob ? <Terminal logs={logs} /> : <div style={{ padding: '2rem', textAlign: 'center', color: '#666', height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>No job running. Waiting for queue...</div>}
                    </div>
                </div>
            </div>

            <div style={{ display: 'flex', gap: '2rem', flexWrap: 'wrap' }}>

                {/* Active Queue */}
                <div style={{ flex: 1, minWidth: '400px' }}>
                    <h3 style={{ color: '#3498db' }}>Active Queue (Next to Run)</h3>
                    {queueJobs.map(job => (
                        <div key={job.id} className="card" style={{ borderLeft: job.status === 'running' ? '4px solid #3498db' : '4px solid #f1c40f' }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                                <strong>{job.config.model.type}</strong>
                                <StatusBadge status={job.status} />
                            </div>
                            <div style={{ fontSize: '0.8rem', color: '#aaa', margin: '0.5rem 0' }}>{job.id} - {new Date(job.created_at * 1000).toLocaleString()}</div>
                            <div style={{ display: 'flex', gap: '0.5rem' }}>
                                <button onClick={() => handleStop(job.id)} className="btn-secondary"><Square size={14} /> Stop/Deque</button>
                                <button onClick={() => setViewLogJob(job)} className="btn-secondary"><FileText size={14} /> Logs</button>
                            </div>
                        </div>
                    ))}
                    {queueJobs.length === 0 && <p style={{ color: '#666', fontStyle: 'italic' }}>Queue is empty.</p>}
                </div>

                {/* Job Pool */}
                <div style={{ flex: 1, minWidth: '400px' }}>
                    <h3 style={{ color: '#95a5a6' }}>Job Pool & History</h3>
                    {poolJobs.map(job => (
                        <div key={job.id} className="card" style={{ opacity: 0.8 }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                                <strong>{job.config.model.type}</strong>
                                <StatusBadge status={job.status} />
                            </div>
                            <div style={{ fontSize: '0.8rem', color: '#aaa', margin: '0.5rem 0' }}>{job.id} - {new Date(job.created_at * 1000).toLocaleString()}</div>
                            <div style={{ display: 'flex', gap: '0.5rem' }}>
                                <button onClick={() => handleQueue(job.id)} className="btn-primary" style={{ padding: '4px 8px' }}><Play size={14} /> Enqueue</button>
                                <button onClick={() => setViewLogJob(job)} className="btn-secondary"><FileText size={14} /> Logs</button>
                                <button onClick={() => handleDelete(job.id)} className="btn-danger"><Trash2 size={14} /></button>
                            </div>
                        </div>
                    ))}
                    {poolJobs.length === 0 && <p style={{ color: '#666', fontStyle: 'italic' }}>No jobs in pool.</p>}
                </div>
            </div>
        </div>
    );
};

export default Dashboard;