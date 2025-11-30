import React, { useState, useEffect, useRef } from 'react';
import { useForm, useFieldArray } from 'react-hook-form';
import apiClient from '../api'; // Chemin ajusté
import Terminal from '../components/Terminal'; // Chemin ajusté

function Jobs() {
  const [generatedCmd, setGeneratedCmd] = useState(null);
  const [status, setStatus] = useState('idle');
  const [errorMsg, setErrorMsg] = useState('');
  
  // États pour l'exécution (Phase 2)
  const [isTraining, setIsTraining] = useState(false);
  const [logs, setLogs] = useState([]);
  const wsRef = useRef(null);

  const { register, control, handleSubmit, formState: { errors } } = useForm({
    defaultValues: {
      output_dir: 'output',
      epochs: 10,
      model: {
        type: 'hunyuan-video',
        params: {
            transformer_path: '/path/to/transformer.safetensors',
            vae_path: '/path/to/vae',
            llm_path: '/path/to/llm'
        }
      },
      dataset_config: {
        resolutions: [512],
        enable_ar_bucket: true,
        directories: [
          { path: 'data', num_repeats: 1 }
        ]
      },
      optimizer: { type: 'adamw', lr: 2e-5 }
    }
  });

  const { fields, append, remove } = useFieldArray({
    control,
    name: "dataset_config.directories"
  });

  // --- Gestion WebSocket ---
  const connectWebSocket = () => {
    if (wsRef.current) wsRef.current.close();

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    // Attention: window.location.host inclut le port du frontend, il faut viser le backend
    // Mais votre proxy Vite gère /api, donc on tente de passer par le même host
    const wsUrl = `${protocol}//${window.location.host}/api/ws/logs`;
    
    console.log(`Connecting to WebSocket: ${wsUrl}`);
    const ws = new WebSocket(wsUrl);

    ws.onopen = () => setLogs(prev => [...prev, ">>> CONNECTED TO LOG STREAM <<<"]);
    ws.onmessage = (event) => setLogs(prev => [...prev, event.data]);
    ws.onclose = () => setLogs(prev => [...prev, ">>> CONNECTION CLOSED <<<"]);
    wsRef.current = ws;
  };

  useEffect(() => {
    return () => { if (wsRef.current) wsRef.current.close(); };
  }, []);

  const onSubmit = async (data) => {
    setStatus('loading');
    setErrorMsg('');
    setGeneratedCmd(null);
    
    if (!Array.isArray(data.dataset_config.resolutions)) {
        data.dataset_config.resolutions = [parseInt(data.dataset_config.resolutions)];
    }

    const payload = {
      output_dir: data.output_dir,
      epochs: parseInt(data.epochs),
      save_dtype: 'bfloat16',
      model: { type: data.model.type, params: data.model.params },
      dataset_config: {
        ...data.dataset_config,
        directories: data.dataset_config.directories.map(d => ({ ...d, num_repeats: parseInt(d.num_repeats) }))
      },
      optimizer: data.optimizer,
      adapter: { enabled: false },
      monitoring: { enable_wandb: false },
      evaluation: { eval_every_n_epochs: 1, eval_datasets: [] }
    };

    try {
      const response = await apiClient.post('/generate-config', payload);
      setGeneratedCmd(response.data.command);
      setStatus('success');
    } catch (err) {
      console.error(err);
      setStatus('error');
      setErrorMsg(err.response?.data?.detail || 'Une erreur est survenue.');
    }
  };

  const handleStartTraining = async () => {
      if (!generatedCmd) return;
      try {
          await apiClient.post('/start-training', { command: generatedCmd });
          setIsTraining(true);
          setLogs([]);
          connectWebSocket();
      } catch (err) {
          alert("Failed to start training: " + (err.response?.data?.detail || err.message));
      }
  };

  const handleStopTraining = async () => {
      if (!confirm("Are you sure you want to stop the training process?")) return;
      try {
          await apiClient.post('/stop-training');
          setIsTraining(false);
          setLogs(prev => [...prev, ">>> STOP SIGNAL SENT <<<"]);
      } catch (err) {
          alert("Error stopping training: " + err.message);
      }
  };

  return (
    <div>
      <h2>Create / Edit Job</h2>
      {/* Formulaire de Configuration */}
        <form onSubmit={handleSubmit(onSubmit)} className="config-form">
          <section className="card">
            <h3>Global Settings</h3>
            <div className="form-group">
              <label>Output Directory</label>
              <input {...register("output_dir", { required: true })} placeholder="e.g., output (relative) or /abs/path" />
            </div>
            <div className="form-group">
              <label>Epochs</label>
              <input type="number" {...register("epochs")} />
            </div>
          </section>

          <section className="card">
            <h3>Model Configuration</h3>
            <div className="form-group">
              <label>Model Type</label>
              <select {...register("model.type")}>
                <option value="hunyuan-video">hunyuan-video</option>
                <option value="flux">flux</option>
                <option value="sdxl">sdxl</option>
              </select>
            </div>
            <div className="form-group">
                <label>Transformer Path</label>
                <input {...register("model.params.transformer_path")} placeholder="/path/to/transformer.safetensors" />
            </div>
          </section>

          <section className="card">
            <div className="header-row" style={{display:'flex', justifyContent:'space-between', marginBottom:'1rem'}}>
                <h3>Dataset Directories</h3>
                <button type="button" onClick={() => append({ path: 'data', num_repeats: 1 })} className="btn-secondary">+ Add</button>
            </div>
            {fields.map((field, index) => (
              <div key={field.id} className="dataset-row">
                <div className="form-group flex-grow">
                  <input {...register(`dataset_config.directories.${index}.path`, { required: true })} placeholder="e.g., data (relative) or /abs/path" />
                </div>
                <div className="form-group w-small">
                  <input type="number" {...register(`dataset_config.directories.${index}.num_repeats`)} placeholder="Repeats" />
                </div>
                <button type="button" onClick={() => remove(index)} className="btn-danger">X</button>
              </div>
            ))}
          </section>

          <button type="submit" className="btn-primary" disabled={status === 'loading' || isTraining}>
            {status === 'loading' ? 'Generating...' : 'Generate Configuration'}
          </button>
        </form>

        {/* Zone de Résultat et d'Exécution */}
        {status === 'success' && generatedCmd && (
            <div className="card" style={{marginTop: '2rem', border: '1px solid #2ecc71'}}>
                <h3>Ready to Launch</h3>
                <div style={{background: '#111', padding: '1rem', overflowX: 'auto', marginBottom: '1rem'}}>
                    <code>{generatedCmd}</code>
                </div>
                
                <div className="action-row" style={{ display: 'flex', gap: '1rem' }}>
                    {!isTraining ? (
                        <button onClick={handleStartTraining} className="btn-primary" style={{ backgroundColor: '#2ecc71' }}>
                            ▶ START TRAINING
                        </button>
                    ) : (
                        <button onClick={handleStopTraining} className="btn-danger">
                            ⏹ STOP TRAINING
                        </button>
                    )}
                </div>
            </div>
        )}
        
        {/* Terminal de Logs */}
        {(isTraining || logs.length > 0) && (
            <div className="terminal-section" style={{ marginTop: '2rem' }}>
                <h3>Training Logs</h3>
                <Terminal logs={logs} />
            </div>
        )}
    </div>
  );
}

export default Jobs;