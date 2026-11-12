//! A call's IMBE frames: read from the `.sdr` Trunk Recorder Pro writes beside it (with `captureFrames`), written as the `.dvcf`
//! the engine reads.

use std::path::{Path, PathBuf};

use trunk_core::trunk::frames::{Codec, VoiceFrame};
use trunk_core::trunk::sdr::sdr_frames;

/// A call's `.sdr`: beside its JSON first (the call may be held in the RAM spool), then under the capture folder by its key.
pub fn find(json: &Path, capture_dir: &Path, path: &str) -> Option<PathBuf> {
    [json.with_extension("sdr"), capture_dir.join(format!("{path}.sdr"))].into_iter().find(|f| f.is_file())
}

/// The voice frames in an `.sdr`.
pub fn read(file: &Path) -> Result<Vec<VoiceFrame>, String> {
    std::fs::read(file).map(|b| sdr_frames(&b)).map_err(|e| format!("cannot read {}: {e}", file.display()))
}

/// The widths of an IMBE frame's eight parameters u0..u7, in its 88 information bits.
const WIDTHS: [usize; 8] = [12, 12, 12, 12, 11, 11, 11, 7];

/// An IMBE frame's information bits (0/1, MSB first) as u0..u7.
pub fn params(bits: &[u8]) -> [u32; 8] {
    let mut u = [0u32; 8];
    let mut bits = bits.iter();
    for (p, w) in u.iter_mut().zip(WIDTHS) {
        for b in bits.by_ref().take(w) {
            *p = *p << 1 | u32::from(*b & 1);
        }
    }
    u
}

/// The call's IMBE frames as a `.dvcf`: one SymbolStream v2 CODEC_FRAME each, u7 shifted left one as trunk-recorder's are (the
/// engine reads them as captures). Frames lost on air are left out, as trunk-recorder leaves them out; frame k is stamped
/// k × 20 ms into the call. Also how many frames it holds.
pub fn dvcf(frames: &[VoiceFrame], tg: u32, src: u32, start_us: u64) -> (Vec<u8>, usize) {
    let (mut out, mut n) = (Vec::new(), 0);
    for (k, f) in frames.iter().enumerate() {
        if f.codec != Codec::Imbe || f.erased || f.bits.len() != 88 {
            continue;
        }
        let mut u = params(&f.bits);
        u[7] <<= 1;
        let mut payload = Vec::with_capacity(24 + 32);
        payload.extend(tg.to_le_bytes());
        payload.extend(src.to_le_bytes());
        payload.extend(0u32.to_le_bytes()); // call id
        payload.extend((start_us + k as u64 * 20_000).to_le_bytes());
        payload.extend([0, 8, f.errs.min(255) as u8, 0]); // codec IMBE, 8 parameters, errors, flags
        u.iter().for_each(|v| payload.extend(v.to_le_bytes()));
        out.extend(b"SY");
        out.extend([2, 0x01]);
        out.extend((payload.len() as u32).to_le_bytes());
        out.extend(payload);
        n += 1;
    }
    (out, n)
}

#[cfg(test)]
mod tests {
    use super::*;
    use trunk_core::mbe::Kind;

    fn frame(codec: Codec, bits: Vec<u8>, erased: bool) -> VoiceFrame {
        VoiceFrame { codec, bits, e0: 0, errs: 3, erased, kind: Kind::Voice, tone: None }
    }

    /// `u` packed MSB first into 88 bits.
    fn bits_of(u: [u32; 8]) -> Vec<u8> {
        u.iter().zip(WIDTHS).flat_map(|(v, w)| (0..w).rev().map(move |i| (v >> i & 1) as u8)).collect()
    }

    #[test]
    fn parameters_come_out_of_the_bits_at_their_widths() {
        let u = [0xabc, 0x123, 0xfff, 0x001, 0x7ff, 0x400, 0x2aa, 0x55];
        assert_eq!(params(&bits_of(u)), u);
    }

    #[test]
    fn the_dvcf_keeps_imbe_frames_heard_and_shifts_u7() {
        let u = [0x123, 0x456, 0x789, 0xabc, 0x1ff, 0x2ee, 0x3dd, 0x7f];
        let frames = vec![
            frame(Codec::Imbe, bits_of(u), false),
            frame(Codec::Imbe, bits_of(u), true),
            frame(Codec::Ambe, vec![0; 49], false),
            frame(Codec::Imbe, bits_of(u), false),
        ];
        let (bytes, n) = dvcf(&frames, 4100, 1234, 1_000_000);
        assert_eq!(n, 2);
        assert_eq!(bytes.len(), 2 * (8 + 24 + 32));
        let rec = &bytes[64..128];
        assert_eq!(&rec[..4], b"SY\x02\x01");
        assert_eq!(u32::from_le_bytes(rec[4..8].try_into().unwrap()), 56);
        let p = &rec[8..];
        assert_eq!(u32::from_le_bytes(p[0..4].try_into().unwrap()), 4100);
        assert_eq!(u32::from_le_bytes(p[4..8].try_into().unwrap()), 1234);
        assert_eq!(u64::from_le_bytes(p[12..20].try_into().unwrap()), 1_000_000 + 3 * 20_000);
        assert_eq!(&p[20..24], &[0, 8, 3, 0]);
        let got: Vec<u32> = (0..8).map(|j| u32::from_le_bytes(p[24 + 4 * j..28 + 4 * j].try_into().unwrap())).collect();
        assert_eq!(got[..7], u[..7]);
        assert_eq!(got[7], 0x7f << 1);
    }
}
