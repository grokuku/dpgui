import React, { useEffect, useState } from 'react';
import apiClient from '../api';
import { Activity, Cpu, HardDrive } from 'lucide-react';

const ProgressBar = ({ percent, color = '#646cff', label }) => (
    <div style={{ marginBottom: '1rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.2rem', fontSize: '0.9rem' }}>
            <span>{label}</span>
            <span>{percent}%</span>
        </div>
        <div style={{ height: '8px', background: '#333', borderRadius: '4px', overflow: 'hidden' }}>
            <div style={{ 
                width: `${Math.min(100, Math.max(0, percent))}%`, 
                height: '100%', 
                background: color,
                transition: 'width 0.5s ease' 
            }} />
        </div>
    </div>
);

const Dashboard = () => {
  const [stats, setStats] = useState(null);

  // Polling des stats toutes les 2 secondes
  useEffect(() => {
    const fetchStats = async () => {
        try {
            const res = await apiClient.get('/system-stats');
            setStats(res.data);
        } catch (e) {
            console.error("Failed to fetch stats", e);
        }
    };

    fetchStats(); // Premier appel
    const interval = setInterval(fetchStats, 2000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div>
      <h2>Dashboard</h2>
      
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '1rem', marginBottom: '2rem' }}>
        
        {/* System Monitor Card */}
        <div className="card">
          <div style={{display:'flex', alignItems:'center', gap:'10px', marginBottom:'1rem'}}>
             <Activity size={20} color="#646cff" />
             <h3>System Resources</h3>
          </div>
          
          {stats ? (
             <>
                <ProgressBar percent={stats.cpu} label="CPU Usage" color="#e74c3c" />
                <ProgressBar percent={stats.ram.percent} label="RAM Usage" color="#f1c40f" />
                
                {stats.gpu.length > 0 ? (
                    stats.gpu.map((g, idx) => (
                        <div key={idx} style={{marginTop:'1rem', borderTop:'1px solid #444', paddingTop:'0.5rem'}}>
                            <div style={{fontSize:'0.8rem', color:'#aaa', marginBottom:'0.5rem'}}>GPU {g.id}</div>
                            <ProgressBar percent={g.usage} label="Compute" color="#2ecc71" />
                            <ProgressBar percent={g.vram_percent} label={`VRAM (${(g.vram_used / 1024).toFixed(1)} / ${(g.vram_total / 1024).toFixed(1)} GB)`} color="#9b59b6" />
                        </div>
                    ))
                ) : (
                    <p style={{fontSize: '0.8rem', color: '#666', marginTop:'1rem'}}>No NVIDIA GPU detected.</p>
                )}
             </>
          ) : (
             <p>Loading stats...</p>
          )}
        </div>

        {/* Active Job Placeholder (Future) */}
        <div className="card">
          <div style={{display:'flex', alignItems:'center', gap:'10px', marginBottom:'1rem'}}>
             <Cpu size={20} color="#2ecc71" />
             <h3>Active Job</h3>
          </div>
          <p>No active job running.</p>
        </div>
      </div>

      {/* Queue Placeholder */}
      <div className="card">
        <div style={{display:'flex', alignItems:'center', gap:'10px', marginBottom:'1rem'}}>
            <HardDrive size={20} color="#9b59b6" />
            <h3>Queue Status</h3>
        </div>
        <table style={{ width: '100%', marginTop: '1rem', textAlign: 'left', borderCollapse:'collapse' }}>
            <thead>
                <tr style={{ borderBottom: '1px solid #444', color:'#888', fontSize:'0.9rem' }}>
                    <th style={{padding:'0.5rem'}}>ID</th>
                    <th>Type</th>
                    <th>Status</th>
                    <th>Created</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td colSpan="4" style={{ textAlign: 'center', padding: '2rem', color:'#666' }}>Queue is empty</td>
                </tr>
            </tbody>
        </table>
      </div>
    </div>
  );
};

export default Dashboard;