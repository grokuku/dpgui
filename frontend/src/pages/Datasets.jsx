import React, { useState, useEffect, useRef } from 'react';
import apiClient from '../api';
import { 
    Folder, Image as ImageIcon, FileText, Save, X, 
    AlertCircle, Check, Loader2, Plus, Trash2, Edit2, Copy, 
    Wand2, Scissors, Type, UploadCloud, MousePointer2, Layers, Download, Database
} from 'lucide-react';

// --- Utils ---
const cn = (...classes) => classes.filter(Boolean).join(' ');

// Hook pour le debounce (Auto-save)
function useDebounce(value, delay) {
    const [debouncedValue, setDebouncedValue] = useState(value);
    useEffect(() => {
        const handler = setTimeout(() => setDebouncedValue(value), delay);
        return () => clearTimeout(handler);
    }, [value, delay]);
    return debouncedValue;
}

// --- Components ---

const Modal = ({ title, children, onClose, footer }) => (
    <div style={{
        position: 'fixed', top: 0, left: 0, right: 0, bottom: 0,
        background: 'rgba(0,0,0,0.7)', zIndex: 1000,
        display: 'flex', justifyContent: 'center', alignItems: 'center'
    }}>
        <div className="card" style={{ width: '500px', maxHeight: '90vh', display: 'flex', flexDirection: 'column', padding: '0', overflow: 'hidden' }}>
            <div style={{ padding: '1rem', borderBottom: '1px solid #333', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <h3 style={{ margin: 0 }}>{title}</h3>
                <button onClick={onClose} className="btn-secondary" style={{ padding: '4px' }}><X size={18} /></button>
            </div>
            <div style={{ padding: '1.5rem', overflowY: 'auto' }}>
                {children}
            </div>
            {footer && (
                <div style={{ padding: '1rem', borderTop: '1px solid #333', background: '#222', display: 'flex', justifyContent: 'flex-end', gap: '0.5rem' }}>
                    {footer}
                </div>
            )}
        </div>
    </div>
);

const Datasets = () => {
    // --- State: Data ---
    const [datasets, setDatasets] = useState([]);
    const [currentDataset, setCurrentDataset] = useState(""); // Vide par défaut
    const [images, setImages] = useState([]);
    const [stats, setStats] = useState({ total: 0, tagged: 0 });
    const [loading, setLoading] = useState(false);
    
    // --- State: Selection & Navigation ---
    const [selectedNames, setSelectedNames] = useState(new Set());
    const [lastSelectedName, setLastSelectedName] = useState(null);
    const [focusedName, setFocusedName] = useState(null);

    // --- State: Quick Editor & Auto-Save ---
    const [quickCaption, setQuickCaption] = useState('');
    const [isAutoSave, setIsAutoSave] = useState(true);
    const [saveStatus, setSaveStatus] = useState('idle'); // idle, loading, saving, saved, error
    const debouncedCaption = useDebounce(quickCaption, 1000);

    // --- State: Modals ---
    const [showCreateModal, setShowCreateModal] = useState(false);
    const [showRenameModal, setShowRenameModal] = useState(false);
    const [showTriggerModal, setShowTriggerModal] = useState(false);
    const [showResizeModal, setShowResizeModal] = useState(false);
    const [fullScreenImage, setFullScreenImage] = useState(null);
    
    // --- State: Inputs ---
    const [newDatasetName, setNewDatasetName] = useState('');
    const [triggerWord, setTriggerWord] = useState('');
    const [resizeRes, setResizeRes] = useState(1024);

    // --- State: Drag'n'Drop ---
    const [isDragging, setIsDragging] = useState(false);
    const [uploading, setUploading] = useState(false);

    // --- Refs ---
    const gridRef = useRef(null);
    const imageRefs = useRef({});
    const isCaptionLoadedRef = useRef(false); 

    // ===========================
    // 1. DATA LOADING
    // ===========================

    const loadDatasets = async (selectName = null) => {
        try {
            const res = await apiClient.get('/datasets');
            setDatasets(res.data);
            
            // Si un nom spécifique est demandé (après création/renommage), on le sélectionne
            if (selectName && res.data.includes(selectName)) {
                setCurrentDataset(selectName);
            } 
            // Sinon, on NE sélectionne RIEN par défaut (comportement demandé)
        } catch (e) { console.error(e); }
    };

    const loadImages = async (datasetName) => {
        // Si "None" est sélectionné
        if (!datasetName) {
            setImages([]);
            setStats({ total: 0, tagged: 0 });
            return;
        }
        
        setLoading(true);
        try {
            const res = await apiClient.get(`/datasets/${datasetName}/images`);
            if (res.data.images) {
                setImages(res.data.images);
                setStats({ total: res.data.total, tagged: res.data.tagged });
            }
        } catch (e) { console.error(e); }
        finally { setLoading(false); setSelectedNames(new Set()); setQuickCaption(""); }
    };

    useEffect(() => { loadDatasets(); }, []);
    useEffect(() => { loadImages(currentDataset); }, [currentDataset]);

    // ===========================
    // 2. QUICK EDITOR & AUTO-SAVE
    // ===========================

    useEffect(() => {
        if (selectedNames.size === 1) {
            const imgName = Array.from(selectedNames)[0];
            const img = images.find(i => i.name === imgName);
            if (img) {
                setSaveStatus('loading');
                isCaptionLoadedRef.current = false;
                apiClient.get('/datasets/caption', { params: { path: img.rel_path } })
                    .then(res => {
                        setQuickCaption(res.data.content || "");
                        setSaveStatus('idle');
                        isCaptionLoadedRef.current = true;
                    })
                    .catch(() => {
                        setQuickCaption("Error loading caption.");
                        setSaveStatus('error');
                    });
            }
        } else {
            setQuickCaption("");
            setSaveStatus('idle');
        }
    }, [selectedNames, images]);

    useEffect(() => {
        if (isAutoSave && isCaptionLoadedRef.current && selectedNames.size === 1) {
            const imgName = Array.from(selectedNames)[0];
            const img = images.find(i => i.name === imgName);
            if (img && debouncedCaption !== undefined) {
                handleSaveCaption(img, debouncedCaption, true);
            }
        }
    }, [debouncedCaption]);

    const handleSaveCaption = async (img, content, isAuto = false) => {
        setSaveStatus('saving');
        try {
            await apiClient.post('/datasets/caption', {
                image_path: img.rel_path,
                content: content
            });
            setSaveStatus('saved');
            if (!isAuto) setTimeout(() => setSaveStatus('idle'), 2000);
            setImages(prev => prev.map(i => i.name === img.name ? { ...i, has_caption: !!content.trim() } : i));
        } catch (e) { setSaveStatus('error'); }
    };

    // ===========================
    // 3. KEYBOARD NAVIGATION & SHORTCUTS
    // ===========================

    useEffect(() => {
        const handleKeyDown = (e) => {
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
            if (showCreateModal || showRenameModal || showTriggerModal || showResizeModal || fullScreenImage) return;

            // DELETE SHORTCUT
            if (e.key === 'Delete' || e.key === 'Backspace') {
                if (selectedNames.size > 0) handleDeleteImages();
                return;
            }

            if (!images.length) return;

            const currentIndex = focusedName ? images.findIndex(i => i.name === focusedName) : -1;
            
            // Initial focus
            if (currentIndex === -1 && !focusedName && images.length > 0) {
                 setFocusedName(images[0].name);
                 return;
            }

            let nextIndex = currentIndex;
            const gridEl = gridRef.current;
            const itemEl = imageRefs.current[images[0].name]; 
            let itemsPerRow = 1;
            
            if (gridEl && itemEl) {
                itemsPerRow = Math.floor(gridEl.clientWidth / (itemEl.getBoundingClientRect().width + 16)) || 1;
            }

            if (e.key === 'ArrowRight') nextIndex = Math.min(images.length - 1, currentIndex + 1);
            if (e.key === 'ArrowLeft') nextIndex = Math.max(0, currentIndex - 1);
            if (e.key === 'ArrowDown') nextIndex = Math.min(images.length - 1, currentIndex + itemsPerRow);
            if (e.key === 'ArrowUp') nextIndex = Math.max(0, currentIndex - itemsPerRow);

            if (nextIndex !== currentIndex) {
                e.preventDefault();
                const targetName = images[nextIndex].name;
                setFocusedName(targetName);
                if (imageRefs.current[targetName]) {
                    imageRefs.current[targetName].scrollIntoView({ behavior: 'smooth', block: 'nearest' });
                }
                if (!e.shiftKey && !e.ctrlKey) {
                    setSelectedNames(new Set([targetName]));
                    setLastSelectedName(targetName);
                }
            }
            if (e.key === 'a' && e.ctrlKey) {
                e.preventDefault();
                setSelectedNames(new Set(images.map(i => i.name)));
            }
        };

        window.addEventListener('keydown', handleKeyDown);
        return () => window.removeEventListener('keydown', handleKeyDown);
    }, [images, focusedName, showCreateModal, selectedNames, fullScreenImage]);

    // ===========================
    // 4. ACTIONS & UPLOAD
    // ===========================

    const handleImageClick = (e, imgName, index) => {
        const newSet = new Set(e.ctrlKey ? selectedNames : []);
        if (e.shiftKey && lastSelectedName) {
            const lastIdx = images.findIndex(img => img.name === lastSelectedName);
            const currentIdx = index;
            const start = Math.min(lastIdx, currentIdx);
            const end = Math.max(lastIdx, currentIdx);
            for (let i = start; i <= end; i++) newSet.add(images[i].name);
        } else {
            if (e.ctrlKey) {
                if (newSet.has(imgName)) newSet.delete(imgName);
                else newSet.add(imgName);
            } else {
                newSet.clear(); 
                newSet.add(imgName);
            }
            setLastSelectedName(imgName);
        }
        setSelectedNames(newSet);
        setFocusedName(imgName);
    };

    const handleDrop = async (e) => {
        e.preventDefault();
        setIsDragging(false);
        if (!currentDataset) return;

        const allowedExtensions = ['.jpg', '.jpeg', '.png', '.webp', '.bmp', '.txt'];
        const files = Array.from(e.dataTransfer.files).filter(f => {
            const ext = f.name.toLowerCase().slice(f.name.lastIndexOf('.'));
            return allowedExtensions.includes(ext);
        });
        
        if (files.length === 0) return;

        setUploading(true);
        const formData = new FormData();
        files.forEach(f => formData.append('files', f));

        try {
            await apiClient.post(`/datasets/${currentDataset}/upload`, formData, {
                headers: { 'Content-Type': 'multipart/form-data' }
            });
            loadImages(currentDataset);
        } catch (err) { alert("Upload failed"); }
        finally { setUploading(false); }
    };

    // --- Ops ---
    const getEffectiveTarget = () => selectedNames.size === 0 ? images.map(i => i.name) : Array.from(selectedNames);

    const handleDeleteImages = async () => {
        const targets = Array.from(selectedNames);
        if (!targets.length) return;
        if (!confirm(`Are you sure you want to delete ${targets.length} image(s) and their captions?`)) return;

        try {
            await apiClient.post('/datasets/delete_images', {
                dataset: currentDataset,
                images: targets
            });
            loadImages(currentDataset);
        } catch (e) { alert("Failed to delete images"); }
    };

    const handleExport = () => {
        if (!currentDataset) return;
        window.open(`${apiClient.defaults.baseURL}/datasets/${currentDataset}/export`, '_blank');
    };

    const handleBatchTrigger = async () => {
        try { await apiClient.post('/datasets/batch', { dataset: currentDataset, images: getEffectiveTarget(), action: 'trigger_word', payload: triggerWord }); setShowTriggerModal(false); setTriggerWord(''); loadImages(currentDataset); } catch (e) { alert("Error"); }
    };
    const handleBatchResize = async () => {
        try { setLoading(true); await apiClient.post('/datasets/batch', { dataset: currentDataset, images: getEffectiveTarget(), action: 'resize', payload: resizeRes.toString() }); setShowResizeModal(false); loadImages(currentDataset); } catch (e) { alert("Error"); setLoading(false); }
    };

    // --- Dataset Ops ---
    const handleCreate = async () => { 
        if(newDatasetName.trim()){ 
            await apiClient.post('/datasets/create', {name:newDatasetName}); 
            // Load and SELECT the new one
            loadDatasets(newDatasetName); 
            setShowCreateModal(false); 
        }
    };
    
    const handleDelete = async () => { 
        if(confirm(`Delete dataset '${currentDataset}'?`)){ 
            await apiClient.post('/datasets/delete', {name:currentDataset}); 
            // Retour à "None"
            setCurrentDataset("");
            loadDatasets(); 
        }
    };
    
    const handleRename = async () => { if(newDatasetName.trim()){ await apiClient.post('/datasets/rename', {name:currentDataset, new_name:newDatasetName}); loadDatasets(newDatasetName); setShowRenameModal(false); }};
    const handleClone = async () => { const n = prompt("Name:", `${currentDataset}_copy`); if(n) { await apiClient.post('/datasets/clone', {name:currentDataset, new_name:n}); loadDatasets(n); }};

    // ===========================
    // RENDER
    // ===========================

    return (
        <div 
            style={{ height: '100%', display: 'flex', flexDirection: 'column', position: 'relative' }}
            onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
            onDragLeave={() => setIsDragging(false)}
            onDrop={handleDrop}
        >
            {/* Global Drop Overlay */}
            {isDragging && currentDataset && (
                <div style={{ position: 'absolute', inset: 0, zIndex: 9999, background: 'rgba(100, 108, 255, 0.2)', border: '4px dashed #646cff', display: 'flex', justifyContent: 'center', alignItems: 'center', pointerEvents: 'none' }}>
                    <div style={{ background: '#1a1a1a', padding: '2rem', borderRadius: '8px', display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                        <UploadCloud size={64} color="#646cff" />
                        <h2>Drop files to upload</h2>
                    </div>
                </div>
            )}
            
            {/* Upload Spinner */}
            {uploading && (
                <div style={{ position: 'absolute', inset: 0, zIndex: 9999, background: 'rgba(0,0,0,0.8)', display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center' }}>
                    <Loader2 className="spin" size={64} color="#646cff" />
                    <h3>Uploading...</h3>
                </div>
            )}

            {/* --- HEADER --- */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
                <div style={{ display: 'flex', gap: '1rem', alignItems: 'center' }}>
                    <h2 style={{ margin: 0 }}>Dataset:</h2>
                    <select 
                        value={currentDataset || ''} 
                        onChange={(e) => setCurrentDataset(e.target.value)} 
                        style={{ fontSize: '1.1rem', padding: '0.4rem', minWidth: '200px' }}
                    >
                        <option value="">-- Select Dataset --</option>
                        {datasets.map(d => <option key={d} value={d}>{d}</option>)}
                    </select>
                    <button onClick={() => setShowCreateModal(true)} className="btn-primary" title="Create New"><Plus size={16} /></button>
                </div>
                <div style={{ display: 'flex', gap: '0.5rem' }}>
                    <button onClick={handleExport} className="btn-secondary" disabled={!currentDataset} title="Download ZIP"><Download size={16} /></button>
                    <button onClick={() => { setNewDatasetName(currentDataset); setShowRenameModal(true); }} className="btn-secondary" disabled={!currentDataset} title="Rename"><Edit2 size={16} /></button>
                    <button onClick={handleClone} className="btn-secondary" disabled={!currentDataset} title="Clone"><Copy size={16} /></button>
                    <button onClick={handleDelete} className="btn-danger" disabled={!currentDataset} title="Delete Dataset"><Trash2 size={16} /></button>
                </div>
            </div>

            {/* --- TOOLBAR --- */}
            <div className="card" style={{ padding: '0.8rem', display: 'flex', gap: '1rem', alignItems: 'center', flexWrap: 'wrap' }}>
                <span style={{ color: '#aaa', fontSize: '0.9rem', marginRight: 'auto' }}>
                    {selectedNames.size > 0 ? `${selectedNames.size} selected` : 'No selection (Apply to All)'}
                </span>
                <button onClick={handleDeleteImages} className="btn-danger" disabled={selectedNames.size === 0} title="Delete selected images"><Trash2 size={16} /></button>
                <div style={{ width: '1px', height: '20px', background: '#444', margin: '0 0.5rem' }} />
                <button onClick={() => setShowTriggerModal(true)} className="btn-secondary" disabled={!currentDataset}><Type size={16} style={{marginRight: 6}} /> Trigger Word</button>
                <button onClick={() => setShowResizeModal(true)} className="btn-secondary" disabled={!currentDataset}><Scissors size={16} style={{marginRight: 6}} /> Resize</button>
                <button onClick={() => alert("Coming soon")} className="btn-secondary" disabled={!currentDataset}><Wand2 size={16} style={{marginRight: 6}} /> Auto-Tag</button>
                <div style={{ width: '1px', height: '20px', background: '#444', margin: '0 0.5rem' }} />
                <button className="btn-primary" onClick={() => document.getElementById('upload-input').click()} disabled={!currentDataset}>
                    <UploadCloud size={16} style={{marginRight: 6}} /> Add Files
                </button>
                <input id="upload-input" type="file" multiple accept=".jpg,.jpeg,.png,.webp,.bmp,.txt" style={{ display: 'none' }} onChange={(e) => { const evt = { preventDefault:()=>{}, dataTransfer: { files: e.target.files } }; handleDrop(evt); }} />
            </div>

            {/* --- MAIN CONTENT --- */}
            <div style={{ flex: 1, display: 'flex', overflow: 'hidden', gap: '1rem' }}>
                
                {/* --- GRID --- */}
                <div ref={gridRef} className="card" style={{ flex: 1, overflowY: 'auto', position: 'relative', display: 'block' }}>
                    
                    {/* EMPTY STATE: NO DATASET SELECTED */}
                    {!currentDataset && (
                         <div style={{ height: '100%', display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center', color: '#666' }}>
                            <Database size={64} style={{ marginBottom: '1rem', opacity: 0.5 }} />
                            <h3>Select a dataset</h3>
                            <p>Choose a dataset from the dropdown above to view images.</p>
                        </div>
                    )}

                    {/* EMPTY STATE: DATASET SELECTED BUT EMPTY */}
                    {currentDataset && images.length === 0 && !loading && (
                        <div style={{ height: '100%', display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center', color: '#666' }}>
                            <UploadCloud size={48} style={{ marginBottom: '1rem', opacity: 0.5 }} />
                            <p>Drag images & text files here</p>
                        </div>
                    )}
                    
                    {/* IMAGES GRID */}
                    {currentDataset && (
                        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(140px, 1fr))', gap: '1rem', padding: '0.5rem' }}>
                            {images.map((img, index) => {
                                const isSelected = selectedNames.has(img.name);
                                const isFocused = focusedName === img.name;
                                return (
                                    <div 
                                        key={img.name}
                                        ref={el => imageRefs.current[img.name] = el}
                                        onClick={(e) => handleImageClick(e, img.name, index)}
                                        onDoubleClick={() => setFullScreenImage(img)}
                                        style={{
                                            position: 'relative', borderRadius: '6px', overflow: 'hidden',
                                            border: isSelected ? '2px solid #646cff' : (isFocused ? '2px solid #444' : '2px solid transparent'),
                                            background: isSelected ? '#333' : '#1a1a1a', cursor: 'pointer', transition: 'transform 0.1s'
                                        }}
                                    >
                                        <div style={{ aspectRatio: '1/1', background: '#000', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                                            <img src={`/api/datasets/thumbnail?path=${encodeURIComponent(img.rel_path)}`} loading="lazy" style={{ maxWidth: '100%', maxHeight: '100%', objectFit: 'contain', opacity: isSelected ? 1 : 0.8 }} />
                                        </div>
                                        <div style={{ position: 'absolute', top: 5, right: 5, background: img.has_caption ? '#2ecc71' : '#e67e22', borderRadius: '50%', padding: 3 }}>
                                            {img.has_caption ? <Check size={10} color="white"/> : <FileText size={10} color="white"/>}
                                        </div>
                                        <div style={{ padding: '4px 8px', fontSize: '0.75rem', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis', color: isSelected ? '#fff' : '#aaa' }}>{img.name}</div>
                                    </div>
                                );
                            })}
                        </div>
                    )}
                </div>

                {/* --- SIDEBAR --- */}
                <div className="card" style={{ width: '350px', display: 'flex', flexDirection: 'column', padding: '0', overflow: 'hidden', flexShrink: 0 }}>
                    {selectedNames.size === 1 ? (
                        <>
                            <div style={{ padding: '1rem', borderBottom: '1px solid #333', display: 'flex', justifyContent: 'space-between', alignItems: 'center', background: '#222' }}>
                                <strong style={{fontSize:'0.9rem', whiteSpace:'nowrap', overflow:'hidden', textOverflow:'ellipsis', maxWidth:'200px'}}>
                                    {Array.from(selectedNames)[0]}
                                </strong>
                                <button onClick={() => setSelectedNames(new Set())} className="btn-secondary" style={{padding:4}}><X size={16}/></button>
                            </div>
                            <div 
                                style={{ flex: '0 0 200px', background: '#000', display: 'flex', alignItems: 'center', justifyContent: 'center', borderBottom: '1px solid #333', cursor: 'zoom-in' }}
                                onClick={() => {
                                    const imgName = Array.from(selectedNames)[0];
                                    const img = images.find(i => i.name === imgName);
                                    if(img) setFullScreenImage(img);
                                }}
                            >
                                 {(() => {
                                    const imgName = Array.from(selectedNames)[0];
                                    const img = images.find(i => i.name === imgName);
                                    return img ? (
                                        <img src={`/api/datasets/image_raw?path=${encodeURIComponent(img.rel_path)}`} style={{maxWidth:'100%', maxHeight:'100%', objectFit:'contain'}} />
                                    ) : null;
                                 })()}
                            </div>
                            <div style={{ padding: '0.5rem 1rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderBottom: '1px solid #333', fontSize: '0.8rem' }}>
                                <label style={{ display: 'flex', alignItems: 'center', gap: '6px', cursor: 'pointer' }}>
                                    <input type="checkbox" checked={isAutoSave} onChange={(e) => setIsAutoSave(e.target.checked)} />
                                    Auto-Save
                                </label>
                                {saveStatus === 'saving' && <span style={{color:'#f1c40f'}}>Saving...</span>}
                                {saveStatus === 'saved' && <span style={{color:'#2ecc71'}}>Saved</span>}
                                {saveStatus === 'error' && <span style={{color:'#e74c3c'}}>Error</span>}
                            </div>
                            <div style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
                                <textarea 
                                    value={quickCaption}
                                    onChange={(e) => setQuickCaption(e.target.value)}
                                    placeholder="Enter tags here..."
                                    style={{ flex: 1, width: '100%', resize: 'none', background: '#1a1a1a', color: '#eee', border: 'none', padding: '1rem', fontFamily: 'monospace', fontSize: '0.9rem', outline: 'none' }}
                                />
                            </div>
                            <div style={{ padding: '1rem', borderTop: '1px solid #333' }}>
                                 <button className="btn-primary" style={{width:'100%'}} onClick={() => {
                                     const imgName = Array.from(selectedNames)[0];
                                     const img = images.find(i => i.name === imgName);
                                     handleSaveCaption(img, quickCaption);
                                 }}>
                                    <Save size={16} style={{marginRight:6}} /> Save Caption
                                 </button>
                            </div>
                        </>
                    ) : selectedNames.size > 1 ? (
                        <div style={{ height: '100%', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', color: '#666', textAlign: 'center', padding: '2rem' }}>
                            <Layers size={48} style={{ marginBottom: '1rem', opacity: 0.5 }} />
                            <h3>{selectedNames.size} items selected</h3>
                            <p style={{fontSize: '0.9rem'}}>Use the toolbar above to apply batch actions.</p>
                        </div>
                    ) : (
                        <div style={{ height: '100%', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', color: '#666', textAlign: 'center', padding: '2rem' }}>
                            <MousePointer2 size={48} style={{ marginBottom: '1rem', opacity: 0.5 }} />
                            <h3>No selection</h3>
                            <p style={{fontSize: '0.9rem'}}>Select an image to edit its caption.</p>
                        </div>
                    )}
                </div>
            </div>

            {/* --- FOOTER --- */}
            <div style={{ marginTop: '0.5rem', display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', color: '#666', padding: '0 1rem' }}>
                <span>Total: <strong>{stats.total}</strong></span>
                <span>Tagged: <strong style={{ color: '#2ecc71' }}>{stats.tagged}</strong> / Untagged: <strong style={{ color: '#e67e22' }}>{stats.total - stats.tagged}</strong></span>
            </div>

            {/* --- FULLSCREEN MODAL --- */}
            {fullScreenImage && (
                <div style={{
                    position: 'fixed', top: 0, left: 0, right: 0, bottom: 0,
                    background: 'rgba(0,0,0,0.95)', zIndex: 2000,
                    display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center'
                }} onClick={() => setFullScreenImage(null)}>
                    <img 
                        src={`/api/datasets/image_raw?path=${encodeURIComponent(fullScreenImage.rel_path)}`} 
                        style={{ maxWidth: '95vw', maxHeight: '95vh', objectFit: 'contain', boxShadow: '0 0 20px rgba(0,0,0,0.5)' }}
                    />
                    <button style={{ position: 'absolute', top: 20, right: 20, background: 'none', border: 'none', color: 'white', cursor: 'pointer' }}>
                        <X size={32} />
                    </button>
                </div>
            )}

            {/* --- MODALS --- */}
            {showCreateModal && <Modal title="Create Dataset" onClose={()=>setShowCreateModal(false)} footer={<button className="btn-primary" onClick={handleCreate}>Create</button>}><div className="form-group"><label>Name</label><input autoFocus type="text" value={newDatasetName} onChange={e=>setNewDatasetName(e.target.value)} onKeyDown={e=>e.key==='Enter'&&handleCreate()}/></div></Modal>}
            {showRenameModal && <Modal title="Rename Dataset" onClose={()=>setShowRenameModal(false)} footer={<button className="btn-primary" onClick={handleRename}>Rename</button>}><div className="form-group"><label>New Name</label><input autoFocus type="text" value={newDatasetName} onChange={e=>setNewDatasetName(e.target.value)} onKeyDown={e=>e.key==='Enter'&&handleRename()}/></div></Modal>}
            {showTriggerModal && <Modal title={`Add Trigger (${selectedNames.size||'All'})`} onClose={()=>setShowTriggerModal(false)} footer={<button className="btn-primary" onClick={handleBatchTrigger}>Add</button>}><div className="form-group"><label>Trigger Word</label><input autoFocus type="text" value={triggerWord} onChange={e=>setTriggerWord(e.target.value)} onKeyDown={e=>e.key==='Enter'&&handleBatchTrigger()}/></div></Modal>}
            {showResizeModal && <Modal title="Batch Resize" onClose={()=>setShowResizeModal(false)} footer={<button className="btn-primary" onClick={handleBatchResize}>Resize</button>}><div className="form-group"><label>Size</label><select value={resizeRes} onChange={e=>setResizeRes(parseInt(e.target.value))}><option value={512}>512px</option><option value={768}>768px</option><option value={1024}>1024px</option></select></div></Modal>}
        </div>
    );
};

export default Datasets;