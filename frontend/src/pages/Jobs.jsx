import React, { useState } from 'react';
import { useForm, useFieldArray } from 'react-hook-form';
import { useNavigate } from 'react-router-dom';
import apiClient from '../api';

function Jobs() {
  const navigate = useNavigate();
  const [generatedCmd, setGeneratedCmd] = useState(null);
  const [configPayload, setConfigPayload] = useState(null);
  const [status, setStatus] = useState('idle');
  const [errorMsg, setErrorMsg] = useState('');
  
  const { register, control, handleSubmit, formState: { errors } } = useForm({
    defaultValues: {
      output_dir: 'output',
      epochs: 10,
      model: {
        type: 'z-image', // Default Z-Image
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

  const onSubmit = async (data) => {
    setStatus('loading');
    setErrorMsg('');
    setGeneratedCmd(null);
    setConfigPayload(null);
    
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
      setConfigPayload(payload);
      setStatus('success');
    } catch (err) {
      console.error(err);
      setStatus('error');
      setErrorMsg(err.response?.data?.detail || 'Une erreur est survenue.');
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
    <div>
      <h2>Create New Job (Z-Image)</h2>
      <form onSubmit={handleSubmit(onSubmit)} className="config-form">
          <section className="card">
            <h3>Global Settings</h3>
            <div className="form-group">
              <label>Output Directory</label>
              <input {...register("output_dir", { required: true })} />
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
              {/* LOCKDOWN: Seul Z-Image est sélectionnable */}
              <select {...register("model.type")}>
                <option value="z-image">Z-Image (Supported)</option>
                <option value="hunyuan-video" disabled>hunyuan-video (Disabled)</option>
                <option value="flux" disabled>flux (Disabled)</option>
                <option value="sdxl" disabled>sdxl (Disabled)</option>
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
                  <input {...register(`dataset_config.directories.${index}.path`, { required: true })} placeholder="path/to/images" />
                </div>
                <div className="form-group w-small">
                  <input type="number" {...register(`dataset_config.directories.${index}.num_repeats`)} placeholder="Repeats" />
                </div>
                <button type="button" onClick={() => remove(index)} className="btn-danger">X</button>
              </div>
            ))}
          </section>

          <button type="submit" className="btn-primary" disabled={status === 'loading'}>
            {status === 'loading' ? 'Generating...' : 'Generate Configuration'}
          </button>
      </form>

      {status === 'success' && generatedCmd && (
          <div className="card" style={{marginTop: '2rem', border: '1px solid #2ecc71'}}>
              <h3>Configuration Validated</h3>
              <div style={{background: '#111', padding: '1rem', overflowX: 'auto', marginBottom: '1rem'}}>
                  <code>{generatedCmd}</code>
              </div>
              <button onClick={handleAddToPool} className="btn-primary" style={{ backgroundColor: '#2ecc71', width: '100%' }}>
                  + SAVE TO JOB POOL
              </button>
          </div>
      )}
    </div>
  );
}

export default Jobs;