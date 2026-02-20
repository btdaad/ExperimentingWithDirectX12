This repository is forked from a release representing the source code for the 2nd lesson in the Learning DirectX 12 series of lessons (https://www.3dgep.com/learning-directx-12-2/).
This project aims to become a **minimal real-time rendering engine**.

Here's what it looks like for the moment :

https://github.com/user-attachments/assets/af7e70a0-d5b1-44bb-8d0c-a25061f9ccfa

You can create a Visual Studio solution by executing `GenerateProjectFiles.bat` file. From then, you can launch the application.
Use A/D to move the camera left/right, W/S to zoom in/out and Q/E to move up/down. By pressing the left mouse button and moving the mouse, you can rotate the camera.

### Features
- [x] Move camera around
- [x] BRDF
- [x] Tone mapping + gamma correction
- [ ] Lights
  - [x] Directional light
  - [ ] Sun
  - [ ] Point light
  - [ ] Area light
- [ ] Textures
- [ ] Import glTF objects <- (work in progress)
- [ ] Sky box
- [ ] IBL
