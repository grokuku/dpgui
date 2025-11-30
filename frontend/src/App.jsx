import React, { useState, useEffect, useRef } from 'react';
import { useForm, useFieldArray } from 'react-hook-form';
import apiClient from './api';
import Terminal from './components/Terminal';
import './App.css';

function App() {
  const [generatedCmd, setGeneratedCmd] = useState(null);
  const [status, setStatus] = useState('idle'); // idle, loading, success, error
  const [errorMsg, setErrorMsg] = useState('');
  
  // États pour l'exécution (Phase 2)
  const [isTraining, setIsTraining] = useState(false);
  const [logs, setLogs] = useState([]);
  const wsRef = useRef(null);

  // Configuration par défaut du formulaire
  const { register, control, handleSubmit, formState: { errors } } = useForm({
    defaultValues: {
      output_dir: 'output', // Chemin relatif (root/output)
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
          { path: 'data', num_repeats: 1 } // Chemin relatif (root/data)
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
    if (wsRef.current) {
        wsRef.current.close();
    }

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/api/ws/logs`;
    
    console.log(`Connecting to WebSocket: ${wsUrl}`);
    const ws = new WebSocket(wsUrl);

    ws.onopen = () => {
        setLogs(prev => [...prev, ">>> CONNECTED TO LOG STREAM <<<"]);
    };

    ws.onmessage = (event) => {
        setLogs(prev => [...prev, event.data]);
    };

    ws.onclose = () => {
        setLogs(prev => [...prev, ">>> CONNECTION CLOSED <<<"]);
    };

    wsRef.current = ws;
  };

  useEffect(() => {
    return () => {
        if (wsRef.current) wsRef.current.close();
    };
  }, []);

  // --- Actions ---

  const onSubmit = async (data) => {
    setStatus('loading');
    setErrorMsg('');
    setGeneratedCmd(null);
    
    // Adaptation des types
    if (!Array.isArray(data.dataset_config.resolutions)) {
        data.dataset_config.resolutions = [parseInt(data.dataset_config.resolutions)];
    }

    const payload = {
      output_dir: data.output_dir,
      epochs: parseInt(data.epochs),
      save_dtype: 'bfloat16',
      model: {
        type: data.model.type,
        params: data.model.params
      },
      dataset_config: {
        ...data.dataset_config,
        directories: data.dataset_config.directories.map(d => ({
            ...d,
            num_repeats: parseInt(d.num_repeats)
        }))
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
          // 1. Démarrer le processus côté serveur
          await apiClient.post('/start-training', { command: generatedCmd });
          setIsTraining(true);
          setLogs([]); // Reset logs
          
          // 2. Se connecter au WebSocket pour écouter
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
    <div className="container">
      <header>
        <h1>Diffusion Pipe GUI</h1>
        <p className="subtitle">Configurator & Launcher</p>
      </header>

      <div className="main-content">
        {/* Formulaire de Configuration */}
        <form onSubmit={handleSubmit(onSubmit)} className="config-form">
          <section className="card">
            <h2>Global Settings</h2>
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
            <h2>Model Configuration</h2>
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
            <div className="header-row">
                <h2>Dataset Directories</h2>
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
            <div className="result-box success">
                <h3>Ready to Launch</h3>
                <div className="code-block"><code>{generatedCmd}</code></div>
                
                <div className="action-row" style={{ display: 'flex', gap: '1rem', marginTop: '1rem' }}>
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
    </div>
  );
}

export default App;