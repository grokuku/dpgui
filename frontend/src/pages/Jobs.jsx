import React, { useState, useEffect } from 'react';
import { useForm, useFieldArray } from 'react-hook-form';
import { useNavigate } from 'react-router-dom';
import apiClient from '../api';
import {
  Save, Play, Terminal, AlertCircle, CheckCircle2,
  HelpCircle, Calculator, Info, Image as ImageIcon,
  Cloud, DownloadCloud, X
} from 'lucide-react';

// --- CONFIGURATION / CONSTANTES ---

const TOOLTIPS = {
  job_name: "The unique name for this training run. A folder with this name will be created in the Output Directory.",
  model_type: "The base architecture to train. 'Z-Image' is currently recommended.",
  model_dtype: "Precision for the model weights. bfloat16 is recommended for Ampere+ GPUs.",
  path_transformer: "Path to the model file (.safetensors) or HuggingFace Repo ID (e.g., 'user/repo').",
  path_vae: "Path to the VAE folder/file. Leave empty if included in the transformer.",
  epochs: "Number of times the model will see the entire dataset.",
  repeats: "How many times an image is repeated per epoch. Higher values = longer epochs.",
  micro_batch_size: "Number of images processed at once per GPU. Lower this if you get OOM (Out Of Memory) errors.",
  grad_accum: "Virtual batch size multiplier. Increases stability but slows down training. Effective Batch = Micro Batch * Grad Accum * GPUs.",
  lr: "Learning Rate. Controls how fast the model learns. Too high = unstable, too low = slow.",
  optimizer: "The mathematical algorithm used for training. AdamW8bit is best for VRAM efficiency.",
  resolutions: "The image sizes used for training. The bucket system will resize images to these targets.",
  eval_prompts: "Prompts used to generate test images during training to monitor progress.",
  eval_every: "How often (in epochs) to generate the test images."
};

const RESOLUTIONS_OPTIONS = [512, 768, 1024, 1280];

const MODEL_TEMPLATES = {
  'z-image': {
    label: 'Z-Image',
    fields: [
      { name: 'transformer_path', label: 'Transformer (Path or HF ID)', required: true, default_repo: "tencent/HunyuanVideo" },
      { name: 'vae_path', label: 'VAE (Path or HF ID)', required: true, default_repo: "tencent/HunyuanVideo" },
      { name: 'llm_path', label: 'LLM (Path or HF ID)', required: true, default_repo: "xtuner/llava-llama-3-8b-v1_1-transformers" }
    ]
  },
  'hunyuan-video': {
    label: 'HunyuanVideo',
    fields: [
      { name: 'transformer_path', label: 'Transformer (Path or HF ID)', required: true, default_repo: "tencent/HunyuanVideo" },
      { name: 'vae_path', label: 'VAE (Path or HF ID)', required: true, default_repo: "tencent/HunyuanVideo" },
      { name: 'llm_path', label: 'LLM (Path or HF ID)', required: true, default_repo: "xtuner/llava-llama-3-8b-v1_1-transformers" }
    ]
  },
  'flux': {
    label: 'Flux.1',
    fields: [
      { name: 'transformer_path', label: 'Transformer Path', required: true, default_repo: "black-forest-labs/FLUX.1-dev", default_filename: "flux1-dev.safetensors" },
      { name: 't5_path', label: 'T5 Encoder Path', required: true, default_repo: "city96/t5-v1_1-xxl-encoder-bf16" },
      { name: 'clip_path', label: 'CLIP Path', required: true, default_repo: "openai/clip-vit-large-patch14" }
    ]
  },
  'ltx-video': {
    label: 'LTX-Video',
    fields: [
      { name: 'transformer_path', label: 'Transformer', required: true, default_repo: "Lightricks/LTX-Video" },
      { name: 'vae_path', label: 'VAE', required: true, default_repo: "Lightricks/LTX-Video" },
      { name: 'text_encoder_path', label: 'Text Encoder (T5)', required: true, default_repo: "Lightricks/LTX-Video" }
    ]
  },
  'sdxl': {
    label: 'SDXL',
    fields: [
      { name: 'checkpoint_path', label: 'Checkpoint File (.safetensors)', required: true, default_repo: "stabilityai/stable-diffusion-xl-base-1.0", default_filename: "sd_xl_base_1.0.safetensors" },
      { name: 'vae_path', label: 'VAE', required: false, default_repo: "madebyollin/sdxl-vae-fp16-fix", default_filename: "sdxl_vae.safetensors" }
    ]
  },
  'wan': {
    label: 'Wan2.1 (Video)',
    fields: [
      { name: 'ckpt_path', label: 'Checkpoint Path', required: true, default_repo: "Wan-AI/Wan2.1-T2V-1.3B" },
      { name: 't5_checkpoint_path', label: 'T5 Path (Optional)', required: false, default_repo: "Wan-AI/Wan2.1-T2V-1.3B" }
    ]
  },
  'lumina_2': {
    label: 'Lumina Image 2.0',
    fields: [
      { name: 'transformer_path', label: 'Transformer Path', required: true, default_repo: "Alpha-VLLM/Lumina-Image-2.0" }
    ]
  },
  'cosmos': {
    label: 'NVIDIA Cosmos',
    fields: [
      { name: 'transformer_path', label: 'Transformer Path', required: true, default_repo: "nvidia/Cosmos-1.0-Diffusion-7B-Text2World" },
      { name: 'vae_path', label: 'VAE Path', required: true, default_repo: "nvidia/Cosmos-1.0-Diffusion-7B-Text2World" }
    ]
  },
  'auraflow': {
    label: 'AuraFlow',
    fields: [
      { name: 'transformer_path', label: 'Transformer Path', required: true, default_repo: "fal/AuraFlow-v0.3" }
    ]
  }
};

const LabelWithTooltip = ({ label, tooltipKey }) => (
  <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '4px' }}>
    <label style={{ margin: 0, fontWeight: 500, color: '#ddd' }}>{label}</label>
    {TOOLTIPS[tooltipKey] && (
      <div className="tooltip-container" style={{ position: 'relative', display: 'flex' }}>
        <HelpCircle size={14} color="#666" style={{ cursor: 'help' }} />
        <span className="tooltip-text">{TOOLTIPS[tooltipKey]}</span>
      </div>
    )}
  </div>
);

// Composant Modal Download Modifié pour supporter 'filename'
const DownloadModal = ({ onClose, onSuccess, initialRepo = '', initialFilename = '' }) => {
  const [repoId, setRepoId] = useState(initialRepo);
  const [filename, setFilename] = useState(initialFilename);
  const [hfToken, setHfToken] = useState('');
  const [status, setStatus] = useState('idle'); // idle, downloading, success, error
  const [downloadedSize, setDownloadedSize] = useState('0 MB');
  const [error, setError] = useState(null);

  // Polling du statut
  useEffect(() => {
    let interval;
    if (status === 'downloading') {
      interval = setInterval(async () => {
        const res = await apiClient.get('/models/status');
        const s = res.data;
        if (s.status === 'completed') {
          setStatus('success');
          // Si on a téléchargé un fichier spécifique, on construit le chemin complet
          let finalPath = s.local_path;
          if (filename && !finalPath.endsWith(filename)) {
             // Basic construction assuming simple folder structure from snapshot_download with allow_patterns
             // Note: snapshot_download returns the folder path usually
             finalPath = `${s.local_path}/${filename}`;
          }
          onSuccess(finalPath);
          clearInterval(interval);
        } else if (s.status === 'error') {
          setStatus('error');
          setError(s.error);
          clearInterval(interval);
        } else {
          // Update size
          setDownloadedSize(s.downloaded_size || "0 MB");
        }
      }, 1000);
    }
    return () => clearInterval(interval);
  }, [status, onSuccess, filename]);

  const startDownload = async () => {
    if (!repoId) return;
    setStatus('downloading');
    setError(null);
    try {
      await apiClient.post('/models/download', {
        repo_id: repoId,
        filename: filename || null, // Pass filename if present
        hf_token: hfToken
      });
    } catch (e) {
      setStatus('error');
      setError(e.response?.data?.detail || "Request failed");
    }
  };

  return (
    <div style={{
      position: 'fixed', top: 0, left: 0, right: 0, bottom: 0,
      background: 'rgba(0,0,0,0.8)', zIndex: 2000,
      display: 'flex', justifyContent: 'center', alignItems: 'center'
    }}>
      <div className="card" style={{ width: '450px', padding: '2rem', border: '1px solid #444', position: 'relative' }}>
        <button onClick={onClose} style={{ position: 'absolute', top: 10, right: 10, background: 'transparent', color: '#666' }}><X size={20} /></button>
        <h3 style={{ marginTop: 0, display: 'flex', alignItems: 'center', gap: '10px' }}>
          <DownloadCloud size={24} color="#646cff" /> Download from HuggingFace
        </h3>

        {status === 'success' ? (
          <div className="result-box success">
            <CheckCircle2 size={20} />
            <div>
              <strong>Download Complete!</strong>
              <p style={{ fontSize: '0.8rem', margin: '5px 0' }}>Path auto-filled.</p>
              <button onClick={onClose} className="btn-primary" style={{ marginTop: '1rem', width: '100%' }}>Close</button>
            </div>
          </div>
        ) : (
          <>
            <div className="form-group">
              <label>Repository ID</label>
              <input
                value={repoId}
                onChange={e => setRepoId(e.target.value)}
                placeholder="user/repo"
                disabled={status === 'downloading'}
              />
            </div>
            <div className="form-group">
              <label>Filename (Optional - for single file)</label>
              <input
                value={filename}
                onChange={e => setFilename(e.target.value)}
                placeholder="model.safetensors"
                disabled={status === 'downloading'}
              />
            </div>
            <div className="form-group">
              <label>HF Token (Optional)</label>
              <input
                type="password"
                value={hfToken}
                onChange={e => setHfToken(e.target.value)}
                placeholder="hf_..."
                disabled={status === 'downloading'}
              />
            </div>

            {status === 'downloading' && (
              <div style={{ margin: '1rem 0', textAlign: 'center', color: '#646cff' }}>
                <div className="spin" style={{ display: 'inline-block', marginBottom: '10px' }}>⏳</div>
                <div>Downloading... {downloadedSize}</div>
                <small style={{ color: '#666', display: 'block' }}>Do not close backend.</small>
              </div>
            )}

            {error && (
              <div className="result-box error" style={{ marginBottom: '1rem' }}>
                <strong>Error:</strong> {error}
              </div>
            )}

            <div style={{ display: 'flex', gap: '1rem', marginTop: '1.5rem' }}>
              <button onClick={onClose} className="btn-secondary" disabled={status === 'downloading'}>Cancel</button>
              <button onClick={startDownload} className="btn-primary" disabled={status === 'downloading' || !repoId} style={{ flex: 1 }}>
                {status === 'downloading' ? 'Downloading...' : 'Start Download'}
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  );
};

function Jobs() {
  const navigate = useNavigate();
  const [datasets, setDatasets] = useState([]);
  const [systemGpus, setSystemGpus] = useState([]);
  const [datasetImageCount, setDatasetImageCount] = useState(0);
  const [localModels, setLocalModels] = useState([]);

  const [generatedCmd, setGeneratedCmd] = useState(null);
  const [configPayload, setConfigPayload] = useState(null);
  const [status, setStatus] = useState('idle');
  const [errorMsg, setErrorMsg] = useState('');

  // State pour la modal download { target, repo, filename }
  const [downloadModalState, setDownloadModalState] = useState(null);

  // Form Setup
  const { register, control, handleSubmit, watch, setValue, getValues, formState: { errors } } = useForm({
    defaultValues: {
      job_name: 'my-training-v1',
      model: { type: 'z-image', dtype: 'bfloat16', params: {} }, 

      // Dataset
      dataset_name: '',
      dataset_resolutions: ['1024'], // Checkbox array
      repeats: 1,

      // Training
      epochs: 10,
      save_every_n_epochs: 4,
      micro_batch_size_per_gpu: 1,
      gradient_accumulation_steps: 1,

      // Optimizer
      optimizer: { type: 'adamw8bit', lr: 2e-5, weight_decay: 0.01 },

      // GPUs
      selected_gpus: [], // indexes

      // Evaluation
      evaluation: {
        prompts: [{ text: "a photo of a cat" }],
        every_n_epochs: 1
      }
    }
  });

  const { fields: promptFields, append: appendPrompt, remove: removePrompt } = useFieldArray({
    control,
    name: "evaluation.prompts"
  });

  // Watchers for Calculator
  const w_dataset = watch("dataset_name");
  const w_repeats = watch("repeats");
  const w_epochs = watch("epochs");
  const w_batch = watch("micro_batch_size_per_gpu");
  const w_grad = watch("gradient_accumulation_steps");
  const w_gpus = watch("selected_gpus");
  const w_model_type = watch("model.type");

  // Load Data
  useEffect(() => {
    apiClient.get('/datasets').then(res => setDatasets(res.data));
    apiClient.get('/system-stats').then(res => {
      if (res.data.gpu && res.data.gpu.length > 0) {
        setSystemGpus(res.data.gpu);
        // Select all GPUs by default
        setValue("selected_gpus", res.data.gpu.map(g => g.id.toString()));
      }
    });
    // Fetch local models for auto-fill
    apiClient.get('/models').then(res => setLocalModels(res.data || []));
  }, []);

  // Update Image Count when dataset changes
  useEffect(() => {
    if (w_dataset) {
      apiClient.get(`/datasets/${w_dataset}/images`).then(res => {
        setDatasetImageCount(res.data.total || 0);
      });
    }
  }, [w_dataset]);

  // AUTO-FILL Logic
  useEffect(() => {
    const template = MODEL_TEMPLATES[w_model_type];
    if (template && template.fields) {
      template.fields.forEach(field => {
        const fieldPath = `model.params.${field.name}`;
        // Seulement si le champ est vide, on essaye de le remplir
        const currentValue = getValues(fieldPath);
        if (!currentValue && field.default_repo) {
          const safeName = field.default_repo.replace(/\//g, "_");
          if (localModels.includes(safeName)) {
            // Found locally!
            setValue(fieldPath, `models/${safeName}`);
          }
        }
      });
    }
  }, [w_model_type, localModels]);

  // --- CALCULATE STEPS ---
  const numGpus = w_gpus ? w_gpus.length : 1;
  const totalSteps = Math.ceil((datasetImageCount * w_repeats * w_epochs) / (Math.max(1, w_batch) * Math.max(1, w_grad) * Math.max(1, numGpus)));

  // --- SUBMIT ---
  const onSubmit = async (data) => {
    setStatus('loading');
    setGeneratedCmd(null);
    setErrorMsg('');

    // Construct Output Path
    const outputDir = `output/${data.job_name}`;

    // Construct Dataset Directories
    const directories = [{
      path: `data/${data.dataset_name}`, // Assuming data root structure
      num_repeats: parseInt(data.repeats)
    }];

    // Construct Config Payload
    const payload = {
      output_dir: outputDir,
      epochs: parseInt(data.epochs),
      save_every_n_epochs: parseInt(data.save_every_n_epochs),
      micro_batch_size_per_gpu: parseInt(data.micro_batch_size_per_gpu),
      gradient_accumulation_steps: parseInt(data.gradient_accumulation_steps),
      save_dtype: 'bfloat16',

      model: {
        type: data.model.type,
        dtype: data.model.dtype, 
        params: data.model.params
      },

      dataset_config: {
        resolutions: data.dataset_resolutions.map(r => parseInt(r)),
        enable_ar_bucket: true,
        directories: directories
      },

      optimizer: {
        ...data.optimizer,
        lr: parseFloat(data.optimizer.lr),
        weight_decay: parseFloat(data.optimizer.weight_decay)
      },

      evaluation: {
        eval_every_n_epochs: parseInt(data.evaluation.every_n_epochs),
        eval_datasets: [] // Placeholder
      },

      monitoring: { enable_wandb: false },
      adapter: { enabled: false }
    };

    try {
      // 1. Generate Config
      const response = await apiClient.post('/generate-config', payload);
      let cmd = response.data.command;

      // 2. Inject GPU selection into command
      if (data.selected_gpus.length > 0) {
        const gpuStr = data.selected_gpus.join(',');
        const numGpus = data.selected_gpus.length;
        // Replace generic deepspeed call
        cmd = `CUDA_VISIBLE_DEVICES=${gpuStr} deepspeed --num_gpus=${numGpus} ` + cmd.split('deepspeed --num_gpus=1 ')[1];
      }

      setGeneratedCmd(cmd);
      setConfigPayload(payload);
      setStatus('success');
    } catch (err) {
      setErrorMsg(err.response?.data?.detail || 'Config generation failed');
      setStatus('error');
    }
  };

  const handleSaveToPool = async () => {
    try {
      await apiClient.post('/jobs', { command: generatedCmd, config: configPayload });
      navigate('/');
    } catch (e) { alert("Error saving job"); }
  };

  return (
    <div style={{ paddingBottom: '4rem' }}>

      {/* HEADER */}
      <div style={{ marginBottom: '1.5rem', borderBottom: '1px solid #333', paddingBottom: '1rem' }}>
        <h2 style={{ margin: 0, marginBottom: '0.5rem' }}>Create Training Job</h2>
        <p style={{ color: '#888', margin: 0, fontSize: '0.9rem' }}>Configure your training session. Hover over <HelpCircle size={12} style={{ display: 'inline' }} /> for details.</p>
      </div>

      <form onSubmit={handleSubmit(onSubmit)}>

        {/* 1. IDENTITY & DATASET */}
        <div className="card" style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2rem' }}>
          {/* Left: Identity */}
          <div>
            <h3 style={{ marginTop: 0, color: '#646cff' }}>1. Project Identity</h3>
            <div className="form-group">
              <LabelWithTooltip label="Job Name (Output Folder)" tooltipKey="job_name" />
              <input {...register("job_name", { required: true })} style={{ fontWeight: 'bold' }} placeholder="my-lora-v1" />
            </div>
            
            {/* ROW: Model Architecture & Dtype */}
            <div style={{ display: 'flex', gap: '1rem' }}>
                <div className="form-group" style={{ flex: 2 }}>
                <LabelWithTooltip label="Model Architecture" tooltipKey="model_type" />
                <select {...register("model.type")}>
                    {Object.entries(MODEL_TEMPLATES).map(([k, v]) => <option key={k} value={k}>{v.label}</option>)}
                </select>
                </div>
                <div className="form-group" style={{ flex: 1 }}>
                    <LabelWithTooltip label="Precision" tooltipKey="model_dtype" />
                    <select {...register("model.dtype")}>
                        <option value="bfloat16">bfloat16</option>
                        <option value="float16">float16</option>
                        <option value="float32">float32</option>
                    </select>
                </div>
            </div>

          </div>

          {/* Right: Dataset */}
          <div>
            <h3 style={{ marginTop: 0, color: '#646cff' }}>2. Source Data</h3>
            <div className="form-group">
              <label>Select Dataset ({datasets.length})</label>
              <select {...register("dataset_name", { required: true })}>
                <option value="">-- Choose Dataset --</option>
                {datasets.map(d => <option key={d} value={d}>{d}</option>)}
              </select>
            </div>

            <div className="form-group">
              <LabelWithTooltip label="Target Resolutions" tooltipKey="resolutions" />
              <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap' }}>
                {RESOLUTIONS_OPTIONS.map(res => (
                  <label key={res} style={{ display: 'flex', alignItems: 'center', gap: '6px', cursor: 'pointer', background: '#333', padding: '4px 8px', borderRadius: '4px' }}>
                    <input type="checkbox" value={res} {...register("dataset_resolutions")} />
                    {res}px
                  </label>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* 2. MODEL CONFIGURATION */}
        <div className="card">
          <h3 style={{ marginTop: 0, color: '#646cff' }}>3. Model Configuration</h3>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(400px, 1fr))', gap: '1rem' }}>
            {MODEL_TEMPLATES[w_model_type]?.fields.map(f => (
              <div key={f.name} className="form-group">
                <LabelWithTooltip label={f.label} tooltipKey={`path_${f.name.split('_')[0]}`} />
                <div style={{ display: 'flex', gap: '8px' }}>
                  <input
                    {...register(`model.params.${f.name}`, { required: f.required })}
                    placeholder={f.default_repo ? `e.g. ${f.default_repo}` : "/path/to/file"}
                    style={{ flex: 1 }}
                  />
                  <button
                    type="button"
                    className="btn-secondary"
                    title={`Download ${f.default_repo || 'model'}`}
                    onClick={() => setDownloadModalState({ 
                        target: `model.params.${f.name}`, 
                        repo: f.default_repo || '',
                        filename: f.default_filename || '' // Pass default filename if available
                    })}
                    style={{ padding: '0 10px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}
                  >
                    <Cloud size={18} />
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* 3. TRAINING PARAMETERS (Main Block) */}
        <div className="card" style={{ position: 'relative' }}>
          <h3 style={{ marginTop: 0, color: '#646cff' }}>4. Training Parameters</h3>

          <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: '2rem' }}>

            {/* Parameters Grid */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
              <div className="form-group">
                <LabelWithTooltip label="Epochs" tooltipKey="epochs" />
                <input type="number" {...register("epochs")} />
              </div>
              <div className="form-group">
                <LabelWithTooltip label="Repeats (per image)" tooltipKey="repeats" />
                <input type="number" {...register("repeats")} />
              </div>

              <div className="form-group">
                <LabelWithTooltip label="Micro Batch Size" tooltipKey="micro_batch_size" />
                <input type="number" {...register("micro_batch_size_per_gpu")} />
              </div>
              <div className="form-group">
                <LabelWithTooltip label="Grad Accumulation" tooltipKey="grad_accum" />
                <input type="number" {...register("gradient_accumulation_steps")} />
              </div>

              <div className="form-group">
                <LabelWithTooltip label="Save Every N Epochs" tooltipKey="epochs" />
                <input type="number" {...register("save_every_n_epochs")} />
              </div>

              <div className="form-group">
                <LabelWithTooltip label="Optimizer Type" tooltipKey="optimizer" />
                <select {...register("optimizer.type")}>
                  <option value="adamw8bit">AdamW 8-bit (Recommended)</option>
                  <option value="adamw">AdamW</option>
                  <option value="prodigy">Prodigy</option>
                </select>
              </div>

              <div className="form-group">
                <LabelWithTooltip label="Learning Rate" tooltipKey="lr" />
                <input type="number" step="1e-6" {...register("optimizer.lr")} />
              </div>
              <div className="form-group">
                <label>Weight Decay</label>
                <input type="number" step="0.001" {...register("optimizer.weight_decay")} />
              </div>
            </div>

            {/* Side Info Panel */}
            <div style={{ background: '#1a1a1a', borderRadius: '8px', padding: '1rem', height: 'fit-content', border: '1px solid #333' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '1rem', color: '#aaa' }}>
                <Calculator size={18} /> <strong>Estimated Steps</strong>
              </div>

              <div style={{ marginBottom: '1rem' }}>
                <div style={{ fontSize: '2rem', fontWeight: 'bold', color: '#fff' }}>
                  {isNaN(totalSteps) ? '-' : totalSteps.toLocaleString()}
                </div>
                <div style={{ fontSize: '0.8rem', color: '#666' }}>Total Steps</div>
              </div>

              <div style={{ fontSize: '0.85rem', color: '#888', display: 'flex', flexDirection: 'column', gap: '4px' }}>
                <div>Images: <span style={{ float: 'right', color: '#ccc' }}>{datasetImageCount}</span></div>
                <div>Total Epochs: <span style={{ float: 'right', color: '#ccc' }}>{w_epochs}</span></div>
                <div style={{ borderTop: '1px solid #333', paddingTop: '4px', marginTop: '4px' }}>
                  Effective Batch: <span style={{ float: 'right', color: '#646cff' }}>{Math.max(1, w_batch) * Math.max(1, w_grad) * numGpus}</span>
                </div>
              </div>

              <div style={{ marginTop: '1.5rem' }}>
                <label style={{ fontWeight: 'bold', display: 'block', marginBottom: '0.5rem' }}>Target GPUs</label>
                {systemGpus.length === 0 ? (
                  <div style={{ color: '#e74c3c', fontSize: '0.8rem' }}>No GPU detected</div>
                ) : (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '5px' }}>
                    {systemGpus.map(gpu => (
                      <label key={gpu.id} style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '0.85rem', cursor: 'pointer' }}>
                        <input type="checkbox" value={gpu.id} {...register("selected_gpus")} />
                        <span>GPU {gpu.id} ({gpu.vram_total / 1024 | 0}GB)</span>
                      </label>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>

        {/* 4. EVALUATION / SAMPLES */}
        <div className="card">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
            <h3 style={{ margin: 0, color: '#646cff' }}>5. Validation Samples</h3>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <label style={{ margin: 0, fontSize: '0.9rem' }}>Generate every</label>
              <input type="number" {...register("evaluation.every_n_epochs")} style={{ width: '60px' }} />
              <span style={{ fontSize: '0.9rem' }}>epochs</span>
            </div>
          </div>

          {promptFields.map((field, index) => (
            <div key={field.id} style={{ display: 'flex', gap: '10px', marginBottom: '10px' }}>
              <div style={{ flex: 1, position: 'relative' }}>
                <ImageIcon size={16} style={{ position: 'absolute', left: 10, top: 12, color: '#666' }} />
                <input
                  {...register(`evaluation.prompts.${index}.text`)}
                  placeholder="Enter a prompt to test model progress..."
                  style={{ paddingLeft: '34px' }}
                />
              </div>
              <button type="button" onClick={() => removePrompt(index)} className="btn-secondary" style={{ padding: '0 10px' }}>
                &times;
              </button>
            </div>
          ))}

          <button type="button" onClick={() => appendPrompt({ text: "" })} className="btn-secondary" style={{ width: '100%', borderStyle: 'dashed', opacity: 0.7 }}>
            + Add Test Prompt
          </button>
        </div>

        {/* 5. ACTIONS */}
        <div style={{ marginTop: '2rem' }}>
          {errorMsg && (
            <div className="result-box error" style={{ marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '10px' }}>
              <AlertCircle color="#e74c3c" /> {errorMsg}
            </div>
          )}

          {!generatedCmd ? (
            <button type="submit" className="btn-primary" disabled={status === 'loading'} style={{ width: '100%', padding: '1rem', fontSize: '1.1rem' }}>
              {status === 'loading' ? 'Validating Configuration...' : 'Generate Configuration'}
            </button>
          ) : (
            <div className="result-box success" style={{ animation: 'fadeIn 0.3s' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '10px', color: '#2ecc71', marginBottom: '1rem' }}>
                <CheckCircle2 size={24} />
                <h3 style={{ margin: 0 }}>Ready to Queue</h3>
              </div>
              <div className="code-block" style={{ maxHeight: '100px', overflowY: 'auto', fontSize: '0.8rem' }}>
                {generatedCmd}
              </div>
              <div style={{ display: 'flex', gap: '1rem', marginTop: '1rem' }}>
                <button type="button" onClick={handleSaveToPool} className="btn-primary" style={{ flex: 1, background: '#2ecc71' }}>
                  <Save size={18} style={{ marginRight: 8 }} /> Save to Job Pool
                </button>
                <button type="button" onClick={() => setGeneratedCmd(null)} className="btn-secondary">
                  Modify Settings
                </button>
              </div>
            </div>
          )}
        </div>

      </form>

      {/* Modal conditionnelle */}
      {downloadModalState && (
        <DownloadModal
          initialRepo={downloadModalState.repo}
          initialFilename={downloadModalState.filename} // Ajout
          onClose={() => setDownloadModalState(null)}
          onSuccess={(path) => {
            setValue(downloadModalState.target, path);
          }}
        />
      )}

      {/* Global Styles for Tooltips */}
      <style>{`
            .tooltip-text {
                visibility: hidden;
                width: 250px;
                background-color: #333;
                color: #fff;
                text-align: center;
                border-radius: 6px;
                padding: 8px;
                position: absolute;
                z-index: 10;
                bottom: 125%;
                left: 50%;
                margin-left: -125px;
                opacity: 0;
                transition: opacity 0.3s;
                font-size: 0.8rem;
                font-weight: normal;
                box-shadow: 0 4px 6px rgba(0,0,0,0.3);
                border: 1px solid #555;
            }
            .tooltip-container:hover .tooltip-text {
                visibility: visible;
                opacity: 1;
            }
        `}</style>
    </div>
  );
}

export default Jobs;