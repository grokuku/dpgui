import React, { useState, useEffect } from 'react';
import { useForm, useFieldArray } from 'react-hook-form';
import { useNavigate } from 'react-router-dom';
import apiClient from '../api';
import {
  Save, Play, Terminal, AlertCircle, CheckCircle2,
  HelpCircle, Calculator, Info, Image as ImageIcon
} from 'lucide-react';

// --- CONFIGURATION / CONSTANTES ---

const TOOLTIPS = {
  job_name: "The unique name for this training run. A folder with this name will be created in the Output Directory.",
  model_type: "The base architecture to train. 'Z-Image' is currently recommended.",
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

const RESOLUTIONS_OPTIONS = [512, 768, 1024];

const MODEL_TEMPLATES = {
  'z-image': {
    label: 'Z-Image',
    fields: [
      { name: 'transformer_path', label: 'Transformer (Path or HF ID)', required: true },
      { name: 'vae_path', label: 'VAE (Path or HF ID)', required: true },
      { name: 'llm_path', label: 'LLM (Path or HF ID)', required: true }
    ]
  },
  'flux': {
    label: 'Flux.1',
    fields: [
      { name: 'transformer_path', label: 'Transformer Path', required: true },
      { name: 't5_path', label: 'T5 Encoder Path', required: true },
      { name: 'clip_path', label: 'CLIP Path', required: true }
    ]
  }
};

// Composant Tooltip simple
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

function Jobs() {
  const navigate = useNavigate();
  const [datasets, setDatasets] = useState([]);
  const [systemGpus, setSystemGpus] = useState([]);
  const [datasetImageCount, setDatasetImageCount] = useState(0);

  const [generatedCmd, setGeneratedCmd] = useState(null);
  const [configPayload, setConfigPayload] = useState(null);
  const [status, setStatus] = useState('idle');
  const [errorMsg, setErrorMsg] = useState('');

  // Form Setup
  const { register, control, handleSubmit, watch, setValue, formState: { errors } } = useForm({
    defaultValues: {
      job_name: 'my-training-v1',
      model: { type: 'z-image', params: {} },

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
  }, []);

  // Update Image Count when dataset changes
  useEffect(() => {
    if (w_dataset) {
      apiClient.get(`/datasets/${w_dataset}/images`).then(res => {
        setDatasetImageCount(res.data.total || 0);
      });
    }
  }, [w_dataset]);

  // --- CALCULATE STEPS ---
  const numGpus = w_gpus ? w_gpus.length : 1;
  const totalSteps = Math.ceil((datasetImageCount * w_repeats * w_epochs) / (Math.max(1, w_batch) * Math.max(1, w_grad) * Math.max(1, numGpus)));

  // --- SUBMIT ---
  const onSubmit = async (data) => {
    setStatus('loading');
    setGeneratedCmd(null);
    setErrorMsg('');

    // Construct Output Path
    // NOTE: This relies on backend having a default output root, we append the name
    const outputDir = `output/${data.job_name}`;

    // Construct Dataset Directories
    // We only support 1 dataset selected from dropdown for now to simplify
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
      save_dtype: 'bfloat16', // Hardcoded default for now

      model: {
        type: data.model.type,
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
        // We need to map prompts to eval_datasets structure required by diff-pipe
        // This is a simplification. Usually eval_datasets points to a TOML.
        // For now, we assume the backend might handle raw prompts or we skip this detail
        // in the generation logic if backend doesn't support raw prompt list yet.
        // Placeholder logic:
        eval_datasets: []
      },

      // Monitoring defaults
      monitoring: { enable_wandb: false },
      adapter: { enabled: false }
    };

    try {
      // 1. Generate Config
      const response = await apiClient.post('/generate-config', payload);
      let cmd = response.data.command;

      // 2. Inject GPU selection into command
      // e.g. "CUDA_VISIBLE_DEVICES=0,1 deepspeed --num_gpus=2 ..."
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
            <div className="form-group">
              <LabelWithTooltip label="Model Architecture" tooltipKey="model_type" />
              <select {...register("model.type")}>
                {Object.entries(MODEL_TEMPLATES).map(([k, v]) => <option key={k} value={k}>{v.label}</option>)}
              </select>
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
              <div style={{ display: 'flex', gap: '1rem' }}>
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
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '1rem' }}>
            {MODEL_TEMPLATES[w_model_type]?.fields.map(f => (
              <div key={f.name} className="form-group">
                <LabelWithTooltip label={f.label} tooltipKey={`path_${f.name.split('_')[0]}`} />
                <input {...register(`model.params.${f.name}`, { required: f.required })} placeholder="/path/to/file or user/repo" />
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