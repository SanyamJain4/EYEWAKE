// main.js
document.addEventListener('DOMContentLoaded', () => {
  const socket = io();

  const statusText = document.getElementById('status-text');
  const statusBubble = document.getElementById('status-bubble');
  const yawnCount = document.getElementById('yawn-count');
  const alertCount = document.getElementById('alert-count');
  const eyeAvg = document.getElementById('eye-avg');
  const log = document.getElementById('log');

  const calibrateBtn = document.getElementById('calibrate-btn');
  const quitBtn = document.getElementById('quit-btn');

  function logMsg(s){
    const t = document.createElement('div');
    t.textContent = `[${new Date().toLocaleTimeString()}] ${s}`;
    log.prepend(t);
  }

  function applyStatus(s){
    const st = (s.status || 'Starting');
    statusText.textContent = st.toUpperCase();
    yawnCount.textContent = s.yawns ?? 0;
    alertCount.textContent = s.alerts ?? 0;
    eyeAvg.textContent = (s.eye_avg || 0).toFixed(2);

    statusBubble.classList.remove('active','tiring','drowsy');
    if(st === 'Active' || st === 'ACTIVE'){
      statusBubble.classList.add('active');
    } else if(st === 'Tiring' || st === 'TIRING'){
      statusBubble.classList.add('tiring');
    } else {
      statusBubble.classList.add('drowsy');
    }
  }

  socket.on('connect', () => {
    logMsg('Connected to server.');
  });

  socket.on('status', (data) => {
    applyStatus(data);
  });

  socket.on('calibration_started', (d) => {
    logMsg('Calibration started (server).');
  });

  socket.on('calibrated', (d) => {
    if(d.error){
      logMsg('Calibration failed: ' + d.error);
    } else {
      logMsg(`Calibration complete. baseline=${d.baseline.toFixed(2)}, threshold=${d.threshold.toFixed(2)}`);
    }
  });

  socket.on('stopped', (d) => {
    logMsg('Server stopped camera: ' + (d.message || ''));
  });

  calibrateBtn.addEventListener('click', () => {
    socket.emit('calibrate', {action:'start'});
    logMsg('Calibration requested.');
  });

  quitBtn.addEventListener('click', () => {
    if(confirm('Stop the capture loop on the server?')){
      socket.emit('quit', {action:'quit'});
      logMsg('Quit requested.');
    }
  });
});
