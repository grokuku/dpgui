import React from 'react';

const Settings = () => {
  return (
    <div>
      <h2>Application Settings</h2>
      <div className="card">
        <div className="form-group">
            <label>Auto-Install Dependencies</label>
            <input type="checkbox" checked readOnly /> Enabled
        </div>
        <div className="form-group">
            <label>Vendor Path</label>
            <input type="text" value="vendor/" disabled />
        </div>
      </div>
    </div>
  );
};

export default Settings;