#pragma once

#include <Camera.h>
#include <Game.h>
#include <Window.h>

#include <tinygltf/tiny_gltf.h>

#include <DirectXMath.h>

using namespace DirectX;

struct Mat
{
    XMMATRIX ModelMatrix;
    XMMATRIX ModelViewMatrix;
    XMMATRIX InverseTransposeModelMatrix;
    XMMATRIX ModelViewProjectionMatrix;
};

struct CameraPositionData
{
    XMVECTOR CameraPos;
    float _padding;
};

class Tutorial2 : public Game
{
public:
    using super = Game;

    Tutorial2(const std::wstring& name, int width, int height, bool vSync = false);
    virtual ~Tutorial2();

    /**
     *  Load content required for the demo.
     */
    virtual bool LoadContent() override;

	bool LoadGLTF(const std::string& filename);
	void LoadGLTFMesh();

    /**
     *  Unload demo specific content that was loaded in LoadContent.
     */
    virtual void UnloadContent() override;
protected:
    /**
     *  Update the game logic.
     */
    virtual void OnUpdate(UpdateEventArgs& e) override;

    /**
     *  Render stuff.
     */
    virtual void OnRender() override;

    /**
     * Invoked by the registered window when a key is pressed
     * while the window has focus.
     */
    virtual void OnKeyPressed(KeyEventArgs& e) override;

    virtual void OnKeyReleased(KeyEventArgs& e) override;

    virtual void OnMouseMoved(MouseMotionEventArgs& e);

    /**
     * Invoked when the mouse wheel is scrolled while the registered window has focus.
     */
    virtual void OnMouseWheel(MouseWheelEventArgs& e) override;


    virtual void OnResize(ResizeEventArgs& e) override; 

private:
    // Helper functions
    // Transition a resource
    void TransitionResource(Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList2> commandList,
        Microsoft::WRL::ComPtr<ID3D12Resource> resource,
        D3D12_RESOURCE_STATES beforeState, D3D12_RESOURCE_STATES afterState);

    // Clear a render target view.
    void ClearRTV(Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList2> commandList,
        D3D12_CPU_DESCRIPTOR_HANDLE rtv, FLOAT* clearColor);

    // Clear the depth of a depth-stencil view.
    void ClearDepth(Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList2> commandList,
        D3D12_CPU_DESCRIPTOR_HANDLE dsv, FLOAT depth = 1.0f );

    // Create a GPU buffer.
    void UpdateBufferResource(Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList2> commandList,
        ID3D12Resource** pDestinationResource, ID3D12Resource** pIntermediateResource,
        size_t numElements, size_t elementSize, const void* bufferData, 
        D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE );

    // Resize the depth buffer to match the size of the client area.
    void ResizeDepthBuffer(int width, int height);

	// Compute and upload matrices for the cube.
    void ComputeAndUploadCubeMatrices(Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList2> commandList);

    uint64_t m_FenceValues[Window::BufferCount] = {};

    // Vertex buffer for the cube.
    Microsoft::WRL::ComPtr<ID3D12Resource> m_VertexBuffer;
    D3D12_VERTEX_BUFFER_VIEW m_VertexBufferView;
    // Index buffer for the cube.
    Microsoft::WRL::ComPtr<ID3D12Resource> m_IndexBuffer;
    D3D12_INDEX_BUFFER_VIEW m_IndexBufferView;
    // Constant buffer for the matrices
	Microsoft::WRL::ComPtr<ID3D12Resource> m_ConstantBuffer;
    uint8_t* m_pCBVDataBegin; // the starting address where the constant buffer will be mapped (from the upload heap to the virtual address space of the app).

    // Depth buffer.
    Microsoft::WRL::ComPtr<ID3D12Resource> m_DepthBuffer;
    // Descriptor heap for depth buffer.
    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> m_DSVHeap;
    // Descriptor heap for CBV
	Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> m_CBVHeap;

    // Root signature
    Microsoft::WRL::ComPtr<ID3D12RootSignature> m_RootSignature;

    // Pipeline state object.
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_PipelineState;

    D3D12_VIEWPORT m_Viewport;
    D3D12_RECT m_ScissorRect;

	Camera m_Camera;
	Mat m_Matrices;
	CameraPositionData m_CameraPositionData;
    struct alignas(16) CameraData
    {
        DirectX::XMVECTOR m_InitialCamPos;
        DirectX::XMVECTOR m_InitialCamRot;
    };
	CameraData* m_pAlignedCameraData;

    // Camera controller
    float m_Forward;
    float m_Backward;
	float m_Left;
	float m_Right;
	float m_Up;
	float m_Down;

    float m_Pitch;
	float m_Yaw;

    int32_t m_PreviousMouseX;
    int32_t m_PreviousMouseY;

    bool m_ContentLoaded;

	tinygltf::Model m_Model;
};