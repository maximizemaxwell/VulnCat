use anyhow::Result;
use candle_core::{Device, Tensor};
use candle_nn::{Module, VarBuilder};

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ModelConfig {
    pub input_dim: usize,
    pub hidden_dim: usize,
    pub latent_dim: usize,
    pub dropout: f64,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            input_dim: 768,
            hidden_dim: 256,
            latent_dim: 64,
            dropout: 0.1,
        }
    }
}

pub struct AutoEncoder {
    encoder: Encoder,
    decoder: Decoder,
    device: Device,
}

struct Encoder {
    fc1: candle_nn::Linear,
    fc2: candle_nn::Linear,
    fc3: candle_nn::Linear,
    dropout: f64,
}

struct Decoder {
    fc1: candle_nn::Linear,
    fc2: candle_nn::Linear,
    fc3: candle_nn::Linear,
    dropout: f64,
}

impl AutoEncoder {
    pub fn new(config: &ModelConfig) -> Result<Self> {
        let device = Device::cuda_if_available(0)?;
        let vb = VarBuilder::zeros(candle_core::DType::F32, &device);

        let encoder = Encoder {
            fc1: candle_nn::linear(config.input_dim, config.hidden_dim, vb.pp("enc_fc1"))?,
            fc2: candle_nn::linear(config.hidden_dim, config.hidden_dim / 2, vb.pp("enc_fc2"))?,
            fc3: candle_nn::linear(config.hidden_dim / 2, config.latent_dim, vb.pp("enc_fc3"))?,
            dropout: config.dropout,
        };

        let decoder = Decoder {
            fc1: candle_nn::linear(config.latent_dim, config.hidden_dim / 2, vb.pp("dec_fc1"))?,
            fc2: candle_nn::linear(config.hidden_dim / 2, config.hidden_dim, vb.pp("dec_fc2"))?,
            fc3: candle_nn::linear(config.hidden_dim, config.input_dim, vb.pp("dec_fc3"))?,
            dropout: config.dropout,
        };

        Ok(Self {
            encoder,
            decoder,
            device,
        })
    }

    pub fn encode(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.encoder.fc1.forward(x)?;
        let x = x.relu()?;
        let x = self.encoder.fc2.forward(&x)?;
        let x = x.relu()?;
        Ok(self.encoder.fc3.forward(&x)?)
    }

    pub fn decode(&self, z: &Tensor) -> Result<Tensor> {
        let x = self.decoder.fc1.forward(z)?;
        let x = x.relu()?;
        let x = self.decoder.fc2.forward(&x)?;
        let x = x.relu()?;
        Ok(self.decoder.fc3.forward(&x)?)
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let z = self.encode(x)?;
        self.decode(&z)
    }

    pub fn device(&self) -> &Device {
        &self.device
    }
}
