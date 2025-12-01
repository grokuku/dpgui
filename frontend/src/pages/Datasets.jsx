import React, { useState, useEffect, useRef, useCallback } from 'react';
import apiClient from '../api';
import { 
    Folder, Image as ImageIcon, FileText, Save, X, 
    AlertCircle, Check, Loader2, Plus, Trash2, Edit2, Copy, 
    Wand2, Scissors, Type, UploadCloud
} from 'lucide-react';

// --- Utils ---
const cn = (...classes) => classes.filter(Boolean).join(' ');

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
    const [currentDataset, setCurrentDataset] = useState(null);
    const [images, setImages] = useState([]);
    const [stats, setStats] = useState({ total: 0, tagged: 0 });
    const [loading, setLoading] = useState(false);
    
    // --- State: Selection & Navigation ---
    const [selectedNames, setSelectedNames] = useState(new Set());
    const [lastSelectedName, setLastSelectedName] = useState(null); // For Shift+Click
    const [focusedName, setFocusedName] = useState(null); // For Keyboard nav

    // --- State: Modals & Dialogs ---
    const [showCreateModal, setShowCreateModal] = useState(false);
    const [showRenameModal, setShowRenameModal] = useState(false);
    const [showTriggerModal, setShowTriggerModal] = useState(false);
    const [showResizeModal, setShowResizeModal] = useState(false);
    
    // --- State: Inputs ---
    const [newDatasetName, setNewDatasetName] = useState('');
    const [triggerWord, setTriggerWord] = useState('');
    const [resizeRes, setResizeRes] = useState(1024);
    
    // --- State: Editor ---
    const [editorImage, setEditorImage] = useState(null);
    const [editorCaption, setEditorCaption] = useState('');
    const [savingCaption, setSavingCaption] = useState(false);

    // --- State: Drag'n'Drop ---
    const [isDragging, setIsDragging] = useState(false);
    const [uploading, setUploading] = useState(false);

    // --- Refs ---
    const gridRef = useRef(null);
    const imageRefs = useRef({});

    // ===========================
    // 1. DATA LOADING
    // ===========================

    const loadDatasets = async () => {
        try {
            const res = await apiClient.get('/datasets');
            setDatasets(res.data);
            if (!currentDataset && res.data.length > 0) {
                setCurrentDataset(res.data[0]);
            }
        } catch (e) { console.error("Load datasets error", e); }
    };

    const loadImages = async (datasetName) => {
        if (!datasetName) return;
        setLoading(true);
        try {
            const res = await apiClient.get(`/datasets/${datasetName}/images`);
            if (res.data.images) {
                setImages(res.data.images);
                setStats({ total: res.data.total, tagged: res.data.tagged });
            }
        } catch (e) { console.error("Load images error", e); }
        finally { setLoading(false); setSelectedNames(new Set()); }
    };

    useEffect(() => { loadDatasets(); }, []);
    useEffect(() => { loadImages(currentDataset); }, [currentDataset]);

    // ===========================
    // 2. SELECTION LOGIC
    // ===========================

    const handleImageClick = (e, imgName, index) => {
        // Prevent opening editor if holding keys
        if (e.ctrlKey || e.shiftKey) {
            e.preventDefault();
        } else {
            // Simple click: select only this one OR open editor?
            // Standard: select only this one. Double click opens editor.
            // Let's stick to: Click = Select, Double Click = Editor (Handled separately)
        }

        const newSet = new Set(e.ctrlKey ? selectedNames : []);
        
        if (e.shiftKey && lastSelectedName) {
            const lastIdx = images.findIndex(img => img.name === lastSelectedName);
            const currentIdx = index;
            const start = Math.min(lastIdx, currentIdx);
            const end = Math.max(lastIdx, currentIdx);
            
            // Add range
            for (let i = start; i <= end; i++) {
                if (e.ctrlKey) {
                    newSet.add(images[i].name); // Add to existing selection
                } else {
                    newSet.add(images[i].name); // New selection is just the range
                }
            }
        } else {
            if (e.ctrlKey) {
                if (newSet.has(imgName)) newSet.delete(imgName);
                else newSet.add(imgName);
            } else {
                newSet.add(imgName);
            }
            setLastSelectedName(imgName);
        }

        setSelectedNames(newSet);
        setFocusedName(imgName);
    };

    // Keyboard Navigation
    useEffect(() => {
        const handleKeyDown = (e) => {
            if (editorImage || showCreateModal || showRenameModal || showTriggerModal) return;
            if (!images.length) return;

            const currentIndex = focusedName ? images.findIndex(i => i.name === focusedName) : -1;
            let nextIndex = currentIndex;

            // Assuming a grid with approx 6 items per row (responsive, but estimation helps)
            // Real calc would need container width, but let's do simple linear for now or +/- 1
            // Improved: Arrow keys navigate linearly for now. Grid nav requires visual calculation.
            
            if (e.key === 'ArrowRight') nextIndex = Math.min(images.length - 1, currentIndex + 1);
            if (e.key === 'ArrowLeft') nextIndex = Math.max(0, currentIndex - 1);
            // Down/Up just jumps by 1 for simplicity in this version, or by 6 if we assume standard width
            // Let's keep it linear navigation for safety
            if (e.key === 'ArrowDown') nextIndex = Math.min(images.length - 1, currentIndex + 1);
            if (e.key === 'ArrowUp') nextIndex = Math.max(0, currentIndex - 1);

            if (nextIndex !== currentIndex) {
                e.preventDefault();
                const targetName = images[nextIndex].name;
                setFocusedName(targetName);
                
                // Scroll into view
                if (imageRefs.current[targetName]) {
                    imageRefs.current[targetName].scrollIntoView({ behavior: 'smooth', block: 'nearest' });
                }

                if (!e.shiftKey && !e.ctrlKey) {
                    // Move selection
                    setSelectedNames(new Set([targetName]));
                    setLastSelectedName(targetName);
                }
            }
            
            // Select All
            if (e.key === 'a' && e.ctrlKey) {
                e.preventDefault();
                setSelectedNames(new Set(images.map(i => i.name)));
            }
        };

        window.addEventListener('keydown', handleKeyDown);
        return () => window.removeEventListener('keydown', handleKeyDown);
    }, [images, focusedName, editorImage, showCreateModal]);

    // ===========================
    // 3. DATASET ACTIONS
    // ===========================

    const handleCreateDataset = async () => {
        if (!newDatasetName.trim()) return;
        try {
            await apiClient.post('/datasets/create', { name: newDatasetName });
            await loadDatasets();
            setCurrentDataset(newDatasetName);
            setShowCreateModal(false);
            setNewDatasetName('');
        } catch (err) { alert(err.response?.data?.detail || "Failed"); }
    };

    const handleDeleteDataset = async () => {
        if (!confirm(`Permanently delete dataset '${currentDataset}'?`)) return;
        try {
            await apiClient.post('/datasets/delete', { name: currentDataset });
            const res = await apiClient.get('/datasets');
            setDatasets(res.data);
            setCurrentDataset(res.data.length ? res.data[0] : null);
        } catch (err) { alert("Failed to delete"); }
    };

    const handleRenameDataset = async () => {
        if (!newDatasetName.trim()) return;
        try {
            await apiClient.post('/datasets/rename', { name: currentDataset, new_name: newDatasetName });
            await loadDatasets();
            setCurrentDataset(newDatasetName);
            setShowRenameModal(false);
        } catch (err) { alert("Failed to rename"); }
    };

    const handleCloneDataset = async () => {
        const cloneName = prompt(`Clone '${currentDataset}' to:`, `${currentDataset}_copy`);
        if (!cloneName) return;
        try {
            await apiClient.post('/datasets/clone', { name: currentDataset, new_name: cloneName });
            await loadDatasets();
            setCurrentDataset(cloneName);
        } catch (err) { alert("Failed to clone"); }
    };

    // ===========================
    // 4. BATCH ACTIONS
    // ===========================

    const getEffectiveTarget = () => {
        // If selection is empty, target ALL images. Else target selected.
        if (selectedNames.size === 0) return images.map(i => i.name);
        return Array.from(selectedNames);
    };

    const handleBatchTrigger = async () => {
        const targets = getEffectiveTarget();
        if (!targets.length) return;
        
        try {
            await apiClient.post('/datasets/batch', {
                dataset: currentDataset,
                images: targets,
                action: 'trigger_word',
                payload: triggerWord
            });
            setShowTriggerModal(false);
            setTriggerWord('');
            loadImages(currentDataset); // Refresh to see caption updates? (Not visible in grid but good practice)
        } catch (err) { alert("Batch operation failed"); }
    };

    const handleBatchResize = async () => {
        const targets = getEffectiveTarget();
        if (!targets.length) return;
        if (!confirm(`Resize ${targets.length} images to ${resizeRes}x${resizeRes}? This overwrites originals.`)) return;

        try {
            setLoading(true);
            await apiClient.post('/datasets/batch', {
                dataset: currentDataset,
                images: targets,
                action: 'resize',
                payload: resizeRes.toString()
            });
            setShowResizeModal(false);
            loadImages(currentDataset); // Force refresh thumbnails
        } catch (err) { alert("Resize failed"); setLoading(false); }
    };

    const handleAutoTag = () => {
        alert("Auto-tagging feature coming in Phase 4.1");
    };

    // ===========================
    // 5. DRAG AND DROP UPLOAD
    // ===========================

    const handleDrop = async (e) => {
        e.preventDefault();
        setIsDragging(false);
        
        if (!currentDataset) return;
        const files = Array.from(e.dataTransfer.files).filter(f => f.type.startsWith('image/'));
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

    // ===========================
    // 6. CAPTION EDITOR
    // ===========================
    const openEditor = async (img) => {
        setEditorImage(img);
        setEditorCaption("Loading...");
        try {
            const res = await apiClient.get('/datasets/caption', { params: { path: img.rel_path } });
            setEditorCaption(res.data.content);
        } catch (e) { setEditorCaption(""); }
    };

    const saveEditorCaption = async () => {
        setSavingCaption(true);
        try {
            await apiClient.post('/datasets/caption', {
                image_path: editorImage.rel_path,
                content: editorCaption
            });
            // Update local state to reflect 'has_caption'
            setImages(prev => prev.map(i => i.name === editorImage.name ? { ...i, has_caption: true } : i));
        } catch (e) { alert("Save failed"); }
        finally { setSavingCaption(false); }
    };

    // ===========================
    // RENDER
    // ===========================

    return (
        <div 
            style={{ height: '100%', display: 'flex', flexDirection: 'column' }}
            onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
            onDragLeave={() => setIsDragging(false)}
            onDrop={handleDrop}
        >
            {/* --- HEADER --- */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
                <div style={{ display: 'flex', gap: '1rem', alignItems: 'center' }}>
                    <h2 style={{ margin: 0, border: 'none' }}>Dataset:</h2>
                    <select 
                        value={currentDataset || ''} 
                        onChange={(e) => setCurrentDataset(e.target.value)}
                        style={{ fontSize: '1.1rem', padding: '0.4rem', minWidth: '200px' }}
                    >
                        {datasets.map(d => <option key={d} value={d}>{d}</option>)}
                    </select>
                    <button onClick={() => setShowCreateModal(true)} className="btn-primary" title="Create New">
                        <Plus size={16} />
                    </button>
                </div>

                <div style={{ display: 'flex', gap: '0.5rem' }}>
                    <button onClick={() => { setNewDatasetName(currentDataset); setShowRenameModal(true); }} className="btn-secondary" disabled={!currentDataset} title="Rename"><Edit2 size={16} /></button>
                    <button onClick={handleCloneDataset} className="btn-secondary" disabled={!currentDataset} title="Clone"><Copy size={16} /></button>
                    <button onClick={handleDeleteDataset} className="btn-danger" disabled={!currentDataset} title="Delete"><Trash2 size={16} /></button>
                </div>
            </div>

            {/* --- TOOLBAR --- */}
            <div className="card" style={{ padding: '0.8rem', display: 'flex', gap: '1rem', alignItems: 'center', flexWrap: 'wrap' }}>
                <span style={{ color: '#aaa', fontSize: '0.9rem', marginRight: 'auto' }}>
                    {selectedNames.size > 0 ? `${selectedNames.size} selected` : 'No selection (Apply to All)'}
                </span>

                <button onClick={() => setShowTriggerModal(true)} className="btn-secondary" disabled={!currentDataset}>
                    <Type size={16} style={{marginRight: 6}} /> Add Trigger Word
                </button>
                <button onClick={() => setShowResizeModal(true)} className="btn-secondary" disabled={!currentDataset}>
                    <Scissors size={16} style={{marginRight: 6}} /> Resize
                </button>
                <button onClick={handleAutoTag} className="btn-secondary" disabled={!currentDataset}>
                    <Wand2 size={16} style={{marginRight: 6}} /> Auto-Tag
                </button>
                <div style={{ width: '1px', height: '20px', background: '#444', margin: '0 0.5rem' }} />
                <button className="btn-primary" onClick={() => document.getElementById('upload-input').click()} disabled={!currentDataset}>
                    <UploadCloud size={16} style={{marginRight: 6}} /> Add Images
                </button>
                <input 
                    id="upload-input" type="file" multiple accept="image/*" style={{ display: 'none' }} 
                    onChange={(e) => {
                        const evt = { preventDefault:()=>{}, dataTransfer: { files: e.target.files } };
                        handleDrop(evt);
                    }}
                />
            </div>

            {/* --- GRID --- */}
            <div 
                ref={gridRef}
                className={cn("card", isDragging ? "drag-active" : "")} 
                style={{ 
                    flex: 1, overflowY: 'auto', position: 'relative', 
                    border: isDragging ? '2px dashed #646cff' : 'none',
                    minHeight: '200px'
                }}
            >
                {uploading && (
                    <div style={{ position: 'absolute', top: 0, left: 0, right: 0, bottom: 0, background: 'rgba(0,0,0,0.7)', zIndex: 10, display: 'flex', justifyContent: 'center', alignItems: 'center', flexDirection: 'column' }}>
                        <Loader2 className="spin" size={48} color="#646cff" />
                        <h3>Uploading...</h3>
                    </div>
                )}
                
                {images.length === 0 && !loading && (
                    <div style={{ height: '100%', display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center', color: '#666' }}>
                        <UploadCloud size={48} style={{ marginBottom: '1rem', opacity: 0.5 }} />
                        <p>Drag and drop images here to start</p>
                    </div>
                )}

                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(140px, 1fr))', gap: '1rem', padding: '0.5rem' }}>
                    {images.map((img, index) => {
                        const isSelected = selectedNames.has(img.name);
                        const isFocused = focusedName === img.name;
                        
                        return (
                            <div 
                                key={img.name}
                                ref={el => imageRefs.current[img.name] = el}
                                onClick={(e) => handleImageClick(e, img.name, index)}
                                onDoubleClick={() => openEditor(img)}
                                style={{
                                    position: 'relative',
                                    borderRadius: '6px',
                                    overflow: 'hidden',
                                    border: isSelected ? '2px solid #646cff' : (isFocused ? '2px solid #444' : '2px solid transparent'),
                                    background: isSelected ? '#333' : '#1a1a1a',
                                    cursor: 'pointer',
                                    transition: 'transform 0.1s'
                                }}
                            >
                                <div style={{ aspectRatio: '1/1', background: '#000', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                                    <img 
                                        src={`/api/datasets/thumbnail?path=${encodeURIComponent(img.rel_path)}`} 
                                        loading="lazy"
                                        style={{ maxWidth: '100%', maxHeight: '100%', objectFit: 'contain', opacity: isSelected ? 1 : 0.8 }}
                                    />
                                </div>
                                
                                {/* Status Icon */}
                                <div style={{ position: 'absolute', top: 5, right: 5, background: img.has_caption ? '#2ecc71' : '#e67e22', borderRadius: '50%', padding: 3, boxShadow: '0 2px 4px rgba(0,0,0,0.5)' }}>
                                    {img.has_caption ? <Check size={10} color="white"/> : <FileText size={10} color="white"/>}
                                </div>
                                
                                <div style={{ padding: '4px 8px', fontSize: '0.75rem', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis', color: isSelected ? '#fff' : '#aaa' }}>
                                    {img.name}
                                </div>
                            </div>
                        );
                    })}
                </div>
            </div>

            {/* --- FOOTER --- */}
            <div style={{ marginTop: '0.5rem', display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', color: '#666', padding: '0 1rem' }}>
                <span>Total Images: <strong>{stats.total}</strong></span>
                <span>Tagged: <strong style={{ color: '#2ecc71' }}>{stats.tagged}</strong> / Untagged: <strong style={{ color: '#e67e22' }}>{stats.total - stats.tagged}</strong></span>
            </div>

            {/* --- MODALS --- */}

            {/* Create Dataset */}
            {showCreateModal && (
                <Modal title="Create New Dataset" onClose={() => setShowCreateModal(false)}
                    footer={<button className="btn-primary" onClick={handleCreateDataset}>Create</button>}
                >
                    <div className="form-group">
                        <label>Dataset Name</label>
                        <input autoFocus type="text" value={newDatasetName} onChange={e => setNewDatasetName(e.target.value)} onKeyDown={e => e.key === 'Enter' && handleCreateDataset()} />
                    </div>
                </Modal>
            )}

            {/* Rename Dataset */}
            {showRenameModal && (
                <Modal title="Rename Dataset" onClose={() => setShowRenameModal(false)}
                    footer={<button className="btn-primary" onClick={handleRenameDataset}>Rename</button>}
                >
                    <div className="form-group">
                        <label>New Name</label>
                        <input autoFocus type="text" value={newDatasetName} onChange={e => setNewDatasetName(e.target.value)} onKeyDown={e => e.key === 'Enter' && handleRenameDataset()} />
                    </div>
                </Modal>
            )}

            {/* Trigger Word */}
            {showTriggerModal && (
                <Modal title={`Add Trigger Word (${selectedNames.size || 'All'} images)`} onClose={() => setShowTriggerModal(false)}
                    footer={<button className="btn-primary" onClick={handleBatchTrigger}>Apply</button>}
                >
                    <p style={{color: '#aaa', fontSize: '0.9rem'}}>Adds text to the beginning of the caption file.</p>
                    <div className="form-group">
                        <label>Trigger Word / Tag</label>
                        <input autoFocus type="text" value={triggerWord} onChange={e => setTriggerWord(e.target.value)} placeholder="e.g. sks, style of..." onKeyDown={e => e.key === 'Enter' && handleBatchTrigger()} />
                    </div>
                </Modal>
            )}

            {/* Resize */}
            {showResizeModal && (
                <Modal title={`Batch Resize (${selectedNames.size || 'All'} images)`} onClose={() => setShowResizeModal(false)}
                    footer={<button className="btn-primary" onClick={handleBatchResize}>Resize Images</button>}
                >
                    <div className="form-group">
                        <label>Target Resolution (Square)</label>
                        <select value={resizeRes} onChange={e => setResizeRes(parseInt(e.target.value))}>
                            <option value={512}>512 x 512</option>
                            <option value={768}>768 x 768</option>
                            <option value={1024}>1024 x 1024</option>
                        </select>
                    </div>
                    <div className="result-box error" style={{marginTop: '1rem'}}>
                        <strong>Warning:</strong> This overwrites original images.
                    </div>
                </Modal>
            )}

            {/* Editor Fullscreen */}
            {editorImage && (
                <div style={{
                    position: 'fixed', top: 0, left: 0, right: 0, bottom: 0,
                    background: 'rgba(0,0,0,0.95)', zIndex: 2000,
                    display: 'flex', flexDirection: 'column'
                }}>
                    <div style={{ padding: '1rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center', background: '#000' }}>
                        <h3 style={{ margin: 0, color: '#fff' }}>{editorImage.name}</h3>
                        <button onClick={() => setEditorImage(null)} className="btn-secondary"><X size={24} /></button>
                    </div>
                    
                    <div style={{ flex: 1, display: 'flex', overflow: 'hidden' }}>
                        <div style={{ flex: 1, display: 'flex', justifyContent: 'center', alignItems: 'center', padding: '1rem' }}>
                            <img 
                                src={`/api/datasets/image_raw?path=${encodeURIComponent(editorImage.rel_path)}`} 
                                style={{ maxWidth: '100%', maxHeight: '100%', objectFit: 'contain', boxShadow: '0 0 20px rgba(0,0,0,0.5)' }}
                            />
                        </div>
                        <div style={{ width: '400px', background: '#1a1a1a', borderLeft: '1px solid #333', display: 'flex', flexDirection: 'column', padding: '1rem' }}>
                            <label style={{ color: '#aaa', marginBottom: '0.5rem' }}>Caption</label>
                            <textarea 
                                value={editorCaption} 
                                onChange={(e) => setEditorCaption(e.target.value)}
                                style={{ flex: 1, background: '#111', color: '#fff', border: '1px solid #333', padding: '1rem', fontFamily: 'monospace', resize: 'none' }}
                            />
                            <button 
                                onClick={saveEditorCaption} 
                                className="btn-primary" 
                                style={{ marginTop: '1rem', display: 'flex', justifyContent: 'center', alignItems: 'center', gap: '8px' }}
                                disabled={savingCaption}
                            >
                                {savingCaption ? <Loader2 className="spin" /> : <Save size={18} />} Save Caption
                            </button>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};

export default Datasets;