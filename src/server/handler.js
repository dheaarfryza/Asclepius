const predictClassification = require('../services/inferenceService');
const crypto = require('crypto');
const storeData = require('../services/storeData');
 
async function postPredictHandler(request, h) {
  const { image } = request.payload;
  const { model } = request.server.app;
 
try {
  const { label, suggestion } = await predictClassification(model, image);
  const id = crypto.randomUUID();
  const createdAt = new Date().toISOString();
 
  const data = {
    "id": id,
    "result": label,
    "suggestion": suggestion,
    "createdAt": createdAt
  }

  await storeData(id, data);
 
  const response = h.response({
    status: 'success',
    message:  'Model is predicted successfully.',
    data
  });
  response.code(201);
  return response;
}
   catch (error) {
  // Menangani error yang mungkin terjadi selama proses prediksi
  console.error('Error during prediction:', error);
    
    const response = h.response({
      status: 'fail',
      message: 'Terjadi kesalahan dalam melakukan prediksi'
    });
    response.code(500);
    return response;
  }
}


module.exports = postPredictHandler;