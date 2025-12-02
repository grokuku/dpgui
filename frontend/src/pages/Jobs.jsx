import React, { useState } from 'react';
import { useForm, useFieldArray, Controller } from 'react-hook-form';
import { useNavigate } from 'react-router-dom';
import apiClient from '../api';
import {
  Save, Play, Settings, Database, Cpu, Activity,
  Layers, AlertCircle, CheckCircle2, Terminal, Plus, Trash2
} from 'lucide-react';

// --- CONFIGURATION DES MODÈLES ---
// Définit quels champs afficher selon le type de modèle sélectionné
const MODEL_TEMPLATES = {
  'z-image': {
    label: 'Z-Image (Experimental)',
    fields: [
      { name: 'transformer_path', label: 'Transformer Path (.safetensors)', required: true },
      { name: 'vae_path', label: 'VAE Path (Folder or File)', required: true },
      { name: 'llm_path', label: 'LLM Path (Folder)', required: true }
    ]
  },
  'hunyuan-video': {
    label: 'Hunyuan Video',
    fields: [
      { name: 'transformer_path', label: 'Transformer Path', required: true },
      { name: 'vae_path', label: 'VAE Path', required: true },
      { name: 'llm_path', label: 'LLM Path', required: true },
      { name: 'clip_path', label: 'CLIP Path', required: true }
    ]
  },
  'flux': {
    label: 'Flux.1 (Dev/Schnell)',
    fields: [
      { name: 'transformer_path', label: 'Transformer Path', required: true },
      { name: 'vae_path', label: 'VAE Path', required: true },
      { name: 'clip_path', label: 'CLIP Path', required: true },
      { name: 't5_path', label: 'T5 Encoder Path', required: true }
    ]
  },
  'sdxl': {
    label: 'SDXL',
    fields: [
      { name: 'vae_path', label: 'VAE Path', required: true },
      { name: 'tokenizer_one_path', label: 'Tokenizer 1', required: true },
      { name: 'tokenizer_two_path', label: 'Tokenizer 2', required: true },
      { name: 'text_encoder_one_path', label: 'Text Encoder 1', required: true },
      { name: 'text_encoder_two_path', label: 'Text Encoder 2', required: true }
    ]
  }
  // Ajoutez d'autres modèles ici selon train.py
};

const TABS = [
  { id: 'general', label: 'General', icon: Settings },
  { id: 'model', label: 'Model', icon: Cpu },
  { id: 'dataset', label: 'Dataset', icon: Database },
  { id: 'optimizer', label: 'Optimizer', icon: Activity },
  { id: 'advanced', label: 'Advanced', icon: Layers },
];

function Jobs() {
  const navigate = useNavigate();
  const [activeTab, setActiveTab] = useState('general');
  const [generatedCmd, setGeneratedCmd] = useState(null);
  const [configPayload, setConfigPayload] = useState(null);
  const [status, setStatus] = useState('idle'); // idle, loading, success, error
  const [errorMsg, setErrorMsg] = useState('');

  const { register, control, handleSubmit, watch, formState: { errors } } = useForm({
    defaultValues: {
      output_dir: 'output/my_training_run',
      epochs: 10,
      micro_batch_size_per_gpu: 1,
      gradient_accumulation_steps: 1,
      save_every_n_epochs: 1,
      save_dtype: 'bfloat16',
      model: {
        type: 'z-image',
        params: {}
      },
      dataset_config: {
        resolutions: "512", // Géré comme string pour l'input, converti à l'envoi
        enable_ar_bucket: true,
        directories: [{ path: 'data', num_repeats: 1 }]
      },
      optimizer: { type: 'adamw', lr: 2e-5, weight_decay: 0.01 },
      adapter: { enabled: false, type: 'lora', rank: 32, dtype: 'bfloat16' },
      monitoring: { enable_wandb: false, wandb_project: 'diffusion-pipe' },
      evaluation: { eval_every_n_epochs: 1 }
    }
  });

  const { fields, append, remove } = useFieldArray({
    control,
    name: "dataset_config.directories"
  });

  // Surveillance du type de modèle pour affichage dynamique
  const selectedModelType = watch("model.type");

  const onSubmit = async (data) => {
    setStatus('loading');
    setErrorMsg('');
    setGeneratedCmd(null);
    setConfigPayload(null);

    // --- Post-Processing des données avant envoi ---

    // 1. Gestion des résolutions (String -> Array[Int] ou Array[List[Int]])
    // Pour l'instant on gère le cas simple : "512" ou "512, 768"
    let resolutions = [512];
    if (typeof data.dataset_config.resolutions === 'string') {
      resolutions = data.dataset_config.resolutions.split(',').map(r => parseInt(r.trim())).filter(n => !isNaN(n));
    } else if (typeof data.dataset_config.resolutions === 'number') {
      resolutions = [data.dataset_config.resolutions];
    }

    const payload = {
      ...data,
      epochs: parseInt(data.epochs),
      micro_batch_size_per_gpu: parseInt(data.micro_batch_size_per_gpu),
      gradient_accumulation_steps: parseInt(data.gradient_accumulation_steps),
      dataset_config: {
        ...data.dataset_config,
        resolutions: resolutions,
        directories: data.dataset_config.directories.map(d => ({
          ...d,
          num_repeats: parseInt(d.num_repeats)
        }))
      },
      // Nettoyage des params du modèle (garder seulement ceux du modèle actif)
      model: {
        type: data.model.type,
        params: data.model.params // On envoie tout, le backend filtrera ou ignorera
      }
    };

    try {
      const response = await apiClient.post('/generate-config', payload);
      setGeneratedCmd(response.data.command);
      setConfigPayload(payload);
      setStatus('success');
    } catch (err) {
      console.error(err);
      setStatus('error');
      setErrorMsg(err.response?.data?.detail || 'An error occurred while generating configuration.');
    }
  };

  const handleAddToPool = async () => {
    if (!generatedCmd || !configPayload) return;
    try {
      await apiClient.post('/jobs', { command: generatedCmd, config: configPayload });
      navigate('/');
    } catch (err) {
      alert("Failed to create job: " + (err.response?.data?.detail || err.message));
    }
  };

  return (
    <div style={{ paddingBottom: '4rem' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '2rem' }}>
        <h2>Create Training Job</h2>
      </div>

      <div style={{ display: 'flex', gap: '2rem', height: '100%' }}>

        {/* --- SIDEBAR TABS --- */}
        <div style={{ width: '200px', display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
          {TABS.map(tab => (
            <button
              key={tab.id}
              type="button"
              onClick={() => setActiveTab(tab.id)}
              className={`nav-link ${activeTab === tab.id ? 'active' : ''}`}
              style={{
                justifyContent: 'flex-start',
                background: activeTab === tab.id ? '#646cff' : 'transparent',
                color: activeTab === tab.id ? 'white' : '#aaa',
                border: 'none', cursor: 'pointer', textAlign: 'left', width: '100%'
              }}
            >
              <tab.icon size={18} /> {tab.label}
            </button>
          ))}
        </div>

        {/* --- FORM AREA --- */}
        <div style={{ flex: 1 }}>
          <form onSubmit={handleSubmit(onSubmit)}>

            {/* --- TAB: GENERAL --- */}
            {activeTab === 'general' && (
              <div className="card">
                <h3>General Settings</h3>
                <div className="form-group">
                  <label>Output Directory</label>
                  <input {...register("output_dir", { required: true })} placeholder="/workspace/output/my-run" />
                  {errors.output_dir && <span style={{ color: 'red' }}>Required</span>}
                </div>

                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
                  <div className="form-group">
                    <label>Epochs</label>
                    <input type="number" {...register("epochs")} />
                  </div>
                  <div className="form-group">
                    <label>Save Every N Epochs</label>
                    <input type="number" {...register("save_every_n_epochs")} />
                  </div>
                </div>

                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
                  <div className="form-group">
                    <label>Micro Batch Size (per GPU)</label>
                    <input type="number" {...register("micro_batch_size_per_gpu")} />
                  </div>
                  <div className="form-group">
                    <label>Gradient Accumulation Steps</label>
                    <input type="number" {...register("gradient_accumulation_steps")} />
                  </div>
                </div>

                <div className="form-group">
                  <label>Save Precision (Dtype)</label>
                  <select {...register("save_dtype")}>
                    <option value="bfloat16">bfloat16 (Recommended)</option>
                    <option value="float16">float16</option>
                    <option value="float32">float32</option>
                  </select>
                </div>
              </div>
            )}

            {/* --- TAB: MODEL --- */}
            {activeTab === 'model' && (
              <div className="card">
                <h3>Model Configuration</h3>
                <div className="form-group">
                  <label>Model Architecture</label>
                  <select {...register("model.type")} style={{ fontSize: '1rem', padding: '0.5rem' }}>
                    {Object.entries(MODEL_TEMPLATES).map(([key, tpl]) => (
                      <option key={key} value={key}>{tpl.label}</option>
                    ))}
                  </select>
                </div>

                <div style={{ marginTop: '1.5rem', padding: '1rem', background: '#1a1a1a', borderRadius: '6px', border: '1px solid #333' }}>
                  <h4 style={{ margin: '0 0 1rem 0', color: '#646cff' }}>Parameters for {MODEL_TEMPLATES[selectedModelType]?.label}</h4>

                  {MODEL_TEMPLATES[selectedModelType]?.fields.map((field) => (
                    <div className="form-group" key={field.name}>
                      <label>
                        {field.label} {field.required && <span style={{ color: 'red' }}>*</span>}
                      </label>
                      <input
                        {...register(`model.params.${field.name}`, { required: field.required })}
                        placeholder={`/path/to/${field.name}`}
                      />
                      {errors.model?.params?.[field.name] && <span style={{ color: '#e74c3c', fontSize: '0.8rem' }}>This path is required</span>}
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* --- TAB: DATASET --- */}
            {activeTab === 'dataset' && (
              <div className="card">
                <h3>Dataset Configuration</h3>

                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginBottom: '1.5rem' }}>
                  <div className="form-group">
                    <label>Training Resolution(s)</label>
                    <input {...register("dataset_config.resolutions")} placeholder="e.g. 512 or 512, 768" />
                    <small style={{ color: '#666' }}>Comma separated for multiple resolutions</small>
                  </div>
                  <div className="form-group" style={{ display: 'flex', alignItems: 'center', paddingTop: '1.5rem' }}>
                    <input type="checkbox" {...register("dataset_config.enable_ar_bucket")} style={{ width: 'auto', marginRight: '10px' }} />
                    <label style={{ marginBottom: 0, cursor: 'pointer' }}>Enable Aspect Ratio Bucketing</label>
                  </div>
                </div>

                <div className="header-row">
                  <h4>Image Directories</h4>
                  <button type="button" onClick={() => append({ path: 'data', num_repeats: 1 })} className="btn-secondary" style={{ fontSize: '0.8rem' }}>
                    <Plus size={14} style={{ marginRight: 4 }} /> Add Folder
                  </button>
                </div>

                {fields.map((field, index) => (
                  <div key={field.id} className="dataset-row" style={{ alignItems: 'flex-start' }}>
                    <div className="form-group flex-grow">
                      <label style={{ fontSize: '0.8rem' }}>Images Path</label>
                      <input {...register(`dataset_config.directories.${index}.path`, { required: true })} placeholder="/path/to/dataset" />
                    </div>
                    <div className="form-group flex-grow">
                      <label style={{ fontSize: '0.8rem' }}>Masks Path (Optional)</label>
                      <input {...register(`dataset_config.directories.${index}.mask_path`)} placeholder="/path/to/masks" />
                    </div>
                    <div className="form-group w-small">
                      <label style={{ fontSize: '0.8rem' }}>Repeats</label>
                      <input type="number" {...register(`dataset_config.directories.${index}.num_repeats`)} min={1} />
                    </div>
                    <button type="button" onClick={() => remove(index)} className="btn-danger" style={{ marginTop: '24px' }}>
                      <Trash2 size={16} />
                    </button>
                  </div>
                ))}
              </div>
            )}

            {/* --- TAB: OPTIMIZER --- */}
            {activeTab === 'optimizer' && (
              <div className="card">
                <h3>Optimizer</h3>
                <div className="form-group">
                  <label>Type</label>
                  <select {...register("optimizer.type")}>
                    <option value="adamw">AdamW (Standard)</option>
                    <option value="adamw8bit">AdamW 8-bit (Less VRAM)</option>
                    <option value="prodigy">Prodigy (Adaptive)</option>
                    <option value="sgd">SGD</option>
                  </select>
                </div>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
                  <div className="form-group">
                    <label>Learning Rate (LR)</label>
                    <input type="number" step="1e-6" {...register("optimizer.lr")} />
                  </div>
                  <div className="form-group">
                    <label>Weight Decay</label>
                    <input type="number" step="0.001" {...register("optimizer.weight_decay")} />
                  </div>
                </div>
              </div>
            )}

            {/* --- TAB: ADVANCED --- */}
            {activeTab === 'advanced' && (
              <div>
                <div className="card">
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '1rem' }}>
                    <h3>Adapter (LoRA)</h3>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                      <input type="checkbox" {...register("adapter.enabled")} />
                      <label style={{ marginBottom: 0 }}>Enable LoRA</label>
                    </div>
                  </div>

                  {watch("adapter.enabled") && (
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
                      <div className="form-group">
                        <label>Rank (Dimension)</label>
                        <input type="number" {...register("adapter.rank")} />
                      </div>
                      <div className="form-group">
                        <label>Dtype</label>
                        <select {...register("adapter.dtype")}>
                          <option value="bfloat16">bfloat16</option>
                          <option value="float16">float16</option>
                          <option value="float32">float32</option>
                        </select>
                      </div>
                      <div className="form-group" style={{ gridColumn: 'span 2' }}>
                        <label>Init from Existing LoRA (Optional)</label>
                        <input {...register("adapter.init_from_existing")} placeholder="/path/to/previous/lora" />
                      </div>
                    </div>
                  )}
                </div>

                <div className="card">
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '1rem' }}>
                    <h3>Monitoring (Weights & Biases)</h3>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                      <input type="checkbox" {...register("monitoring.enable_wandb")} />
                      <label style={{ marginBottom: 0 }}>Enable WandB</label>
                    </div>
                  </div>

                  {watch("monitoring.enable_wandb") && (
                    <>
                      <div className="form-group">
                        <label>API Key (leave empty if logged in via CLI)</label>
                        <input type="password" {...register("monitoring.wandb_api_key")} />
                      </div>
                      <div className="form-group">
                        <label>Project Name</label>
                        <input {...register("monitoring.wandb_run_name")} placeholder="my-awesome-project" />
                      </div>
                    </>
                  )}
                </div>
              </div>
            )}

            {/* --- ACTION BAR --- */}
            <div style={{ marginTop: '2rem', display: 'flex', gap: '1rem' }}>
              <button type="submit" className="btn-primary" disabled={status === 'loading'} style={{ flex: 1 }}>
                {status === 'loading' ? 'Validating...' : 'Generate Configuration'}
              </button>
            </div>
          </form>
        </div>
      </div>

      {/* --- RESULT AREA --- */}
      {errorMsg && (
        <div className="result-box error" style={{ marginTop: '2rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '0.5rem', color: '#e74c3c' }}>
            <AlertCircle size={20} />
            <strong>Validation Failed</strong>
          </div>
          <p>{errorMsg}</p>
        </div>
      )}

      {status === 'success' && generatedCmd && (
        <div className="result-box success" style={{ marginTop: '2rem', animation: 'fadeIn 0.5s' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px', color: '#2ecc71' }}>
              <CheckCircle2 size={24} />
              <h3 style={{ margin: 0, border: 'none' }}>Configuration Ready</h3>
            </div>
            <button onClick={handleAddToPool} className="btn-primary" style={{ backgroundColor: '#2ecc71', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Save size={18} /> Save to Job Pool
            </button>
          </div>

          <p style={{ color: '#ccc', marginBottom: '0.5rem' }}>Run Command Preview:</p>
          <div className="code-block">
            <Terminal size={16} style={{ display: 'inline', marginRight: '8px', verticalAlign: 'middle' }} />
            <code>{generatedCmd}</code>
          </div>
          <p style={{ fontSize: '0.8rem', color: '#888' }}>
            Note: The generated TOML files are located in <code>generated_configs/</code>.
            Clicking "Save to Job Pool" will register this command to be run from the Dashboard.
          </p>
        </div>
      )}
    </div>
  );
}

export default Jobs;