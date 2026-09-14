# Third-party notices

This release preserves existing third-party license files. A root license never replaces component-specific restrictions.

- ISP: `policy/ISP/LICENSE` in Vista, Academic Research and Educational Use License. VISTA builds on ISP's spherical representation and diffusion implementation; the inherited portions retain those restrictions. The same notice is included beside the VISTA backend.
- ACT and DETR: original licenses are preserved in their backend directories.
- Diffusion Policy / DP: see the bundled DP license.
- ViTAL: MIT, copyright Abraham George (2024); see its backend license.
- Image2Sphere: MIT; the original license is retained under `licenses/` in Vista for adapted spherical projection code.
- e3nn fork: training uses BoceHu/e3nn at commit `b02616237eb936ec8586fe7d803c4a3de2588c09`; install it as an external dependency under its upstream license.
- TacEx and libuipc: the simulator integration retains the component licenses in `third_party/TacEx/` in Vista. Simulation assets and compiled libraries are external to the GitHub release.

Paper method/model code and paths have been adapted for VISTA training and real robot deployment. Release maintenance additionally replaces machine-specific entry points and removes generated artifacts. Consult each component's license before reuse.
