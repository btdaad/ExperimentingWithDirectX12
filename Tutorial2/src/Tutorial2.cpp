#include <Tutorial2.h>

#include <Application.h>
#include <CommandQueue.h>
#include <Helpers.h>
#include <SkyBox.h>
#include <Window.h>

#include <wrl.h>
using namespace Microsoft::WRL;

#include <d3dx12.h>
#include <d3dcompiler.h>
#include <DirectXTex.h>

#include <algorithm> // For std::min, std::max, and std::clamp.
#if defined(min)
#undef min
#endif
#if defined(max)
#undef max
#endif

#include <filesystem>
namespace fs = std::filesystem;

using namespace DirectX;

// Vertex data for the displayed object.
struct VertexPosNormTex
{
    XMFLOAT3 Position;
    XMFLOAT3 Normal;
	XMFLOAT2 Uv;
};
std::vector<VertexPosNormTex> g_Vertices;
std::vector<uint32_t> g_Indices;

struct DirectionalLight
{
    XMFLOAT3 LightDirection;
    float _padding;
    XMFLOAT3 LightColor;
    float LightIntensity;
};
DirectionalLight g_DirLight;

struct Material
{
    XMFLOAT3 BaseColor;
    float Roughness;
    float Metallic;
	XMFLOAT3 _padding;
};
Material g_ModelMat;

// =============================================================================
// Constructor / Destructor
// =============================================================================

Tutorial2::Tutorial2(const std::wstring& name, int width, int height, bool vSync)
    : super(name, width, height, vSync)
    , m_ScissorRect(CD3DX12_RECT(0, 0, LONG_MAX, LONG_MAX))
    , m_Viewport(CD3DX12_VIEWPORT(0.0f, 0.0f, static_cast<float>(width), static_cast<float>(height)))
	, m_Matrices{}
    , m_CameraPositionData{}
    , m_Forward(0), m_Backward(0), m_Left(0), m_Right(0), m_Up(0), m_Down(0)
    , m_Pitch(0), m_Yaw(0)
    , m_PreviousMouseX(0), m_PreviousMouseY(0)
    , m_ContentLoaded(false)
{
    XMVECTOR cameraPos = XMVectorSet(0, 0, -10, 1);
    XMVECTOR cameraTarget = XMVectorSet(0, 0, 0, 1);
    XMVECTOR cameraUp = XMVectorSet(0, 1, 0, 0);
    m_Camera.set_LookAt(cameraPos, cameraTarget, cameraUp);
    m_Camera.set_Projection(45.0f, width / static_cast<float>(height), 0.1f, 100.0f);

    m_pAlignedCameraData = static_cast<CameraData*>(_aligned_malloc(sizeof(CameraData), 16));
    if (!m_pAlignedCameraData) throw std::bad_alloc();
    m_pAlignedCameraData->m_InitialCamPos = m_Camera.get_Translation();
    m_pAlignedCameraData->m_InitialCamRot = m_Camera.get_Rotation();

    g_ModelMat.BaseColor = { 0.8f, 0.6f, 0.2f };
	g_ModelMat.Roughness = 0.5f;
	g_ModelMat.Metallic  = 0.0f;

	g_DirLight.LightDirection = { 0.0f, -1.0f, 1.0f };
	g_DirLight.LightColor     = { 1.0f, 1.0f, 1.0f };
	g_DirLight.LightIntensity = 1.0f;
}

Tutorial2::~Tutorial2()
{
    _aligned_free(m_pAlignedCameraData);
}

// =============================================================================
// Helper: Used to create a ID3D12Resource large enough to store the buffer data passed to the function
//         and to create an intermediate buffer that is used to copy the CPU buffer data to the GPU.
// =============================================================================

void Tutorial2::UpdateBufferResource(
    ComPtr<ID3D12GraphicsCommandList2> commandList,
    ID3D12Resource** pDestinationResource,
    ID3D12Resource** pIntermediateResource,
    size_t numElements, size_t elementSize, const void* bufferData,
    D3D12_RESOURCE_FLAGS flags)
{
    auto device = Application::Get().GetDevice();
    size_t bufferSize = numElements * elementSize;

    // Create a committed resource for the GPU resource in a default heap.
    ThrowIfFailed(device->CreateCommittedResource(
        &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
        D3D12_HEAP_FLAG_NONE,
        &CD3DX12_RESOURCE_DESC::Buffer(bufferSize, flags),
        D3D12_RESOURCE_STATE_COMMON,
        nullptr,
        IID_PPV_ARGS(pDestinationResource)));

    // Create an committed resource for the upload.
    if (bufferData)
    {
        ThrowIfFailed(device->CreateCommittedResource(
            &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD),
            D3D12_HEAP_FLAG_NONE,
            &CD3DX12_RESOURCE_DESC::Buffer(bufferSize),
            D3D12_RESOURCE_STATE_GENERIC_READ,
            nullptr,
            IID_PPV_ARGS(pIntermediateResource)));

        D3D12_SUBRESOURCE_DATA subresourceData = {};
        subresourceData.pData       = bufferData;
        subresourceData.RowPitch    = bufferSize;
        subresourceData.SlicePitch  = subresourceData.RowPitch;

        UpdateSubresources(commandList.Get(),
            *pDestinationResource, *pIntermediateResource,
            0, 0, 1, &subresourceData);
    }
}

// ============================================================================
// GLTF loading
// ============================================================================

bool Tutorial2::LoadGLTF(const std::string& filename)
{
    tinygltf::TinyGLTF loader;
    std::string err, warn;
    bool success = loader.LoadASCIIFromFile(&m_Model, &err, &warn, filename);
    if (!warn.empty()) OutputDebugStringA(("GLTF Warning: " + warn).c_str());
    if (!err.empty()) OutputDebugStringA(("GLTF Error: " + err).c_str());
    return success;
}

void Tutorial2::LoadGLTFMesh()
{
    g_Vertices.clear();
    g_Indices.clear();

    for (auto& mesh : m_Model.meshes) // there can be multiple meshes in a glTF file.
    {
        for (auto& primitive : mesh.primitives)
        {
            uint32_t vertexOffset = static_cast<uint32_t>(g_Vertices.size()); // the offset is necessary to correctly index into the vertex buffer

            // POSITIONS
            auto& posAccessor = m_Model.accessors[primitive.attributes.at("POSITION")];
            auto& posView = m_Model.bufferViews[posAccessor.bufferView];
            const float* positions = reinterpret_cast<const float*>(
                m_Model.buffers[posView.buffer].data.data() + posView.byteOffset + posAccessor.byteOffset);

            // NORMALS
            auto& normAccessor = m_Model.accessors[primitive.attributes.at("NORMAL")];
            auto& normView = m_Model.bufferViews[normAccessor.bufferView];
            const float* normals = reinterpret_cast<const float*>(
                m_Model.buffers[normView.buffer].data.data() + normView.byteOffset + normAccessor.byteOffset);

            // TEXCOORDS
            auto& texAccessor = m_Model.accessors[primitive.attributes.at("TEXCOORD_0")];
            auto& texView = m_Model.bufferViews[texAccessor.bufferView];
            const float* texcoords = reinterpret_cast<const float*>(
                m_Model.buffers[texView.buffer].data.data() + texView.byteOffset + texAccessor.byteOffset);

            // Fill vertices
            for (size_t i = 0; i < posAccessor.count; ++i)
            {
                VertexPosNormTex v;
                v.Position = { positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2] };
                v.Normal   = { normals[i * 3],   normals[i * 3 + 1],   normals[i * 3 + 2] };
                v.Uv       = { texcoords[i * 2],   texcoords[i * 2 + 1] };
                g_Vertices.push_back(v);
            }
            
			// INDICES
            auto& idxAccessor = m_Model.accessors[primitive.indices];
            auto& idxView = m_Model.bufferViews[idxAccessor.bufferView];
            const unsigned char* rawIndices = m_Model.buffers[idxView.buffer].data.data()
                + idxView.byteOffset + idxAccessor.byteOffset;

            // Fill indices (shifted by vertexOffset)
            {
                if (idxAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT)
                {
                    const uint16_t* indices = reinterpret_cast<const uint16_t*>(rawIndices);
                    for (size_t i = 0; i < idxAccessor.count; ++i)
                        g_Indices.push_back(static_cast<uint32_t>(indices[i]) + vertexOffset);
                }
                else if (idxAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT)
                {
                    const uint32_t* indices = reinterpret_cast<const uint32_t*>(rawIndices);
                    for (size_t i = 0; i < idxAccessor.count; ++i)
                        g_Indices.push_back(indices[i] + vertexOffset);
                }
            }
        }
    }
}

// ============================================================================
// Texture loading
// Fills m_Texture and m_TextureBuffer.
// Upload pixels, transition to PIXEL_SHADER_RESOURCE and writes SRV into the heap.
// No render, just set up.
// ============================================================================

void Tutorial2::LoadTextureFromFile(
    const std::wstring& fileName,
    Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList2> commandList)
{
    auto device = Application::Get().GetDevice();

    fs::path filePath(fileName);
    if (!fs::exists(filePath)) throw std::exception("File not found.");

    TexMetadata metadata;
    ScratchImage scratchImage;

    if      (filePath.extension() == ".dds") ThrowIfFailed(LoadFromDDSFile(fileName.c_str(), DDS_FLAGS_FORCE_RGB, &metadata, scratchImage));
    else if (filePath.extension() == ".hdr") ThrowIfFailed(LoadFromHDRFile(fileName.c_str(), &metadata, scratchImage));
    else if (filePath.extension() == ".tga") ThrowIfFailed(LoadFromTGAFile(fileName.c_str(), &metadata, scratchImage));
    else                                     ThrowIfFailed(LoadFromWICFile(fileName.c_str(), WIC_FLAGS_FORCE_RGB, &metadata, scratchImage));

    if (metadata.dimension != TEX_DIMENSION_TEXTURE2D)
        throw std::exception("Only 2D textures are supported in this demo.");

    metadata.format = MakeSRGB(metadata.format);

    // Create the final GPU texture resource
    {
        D3D12_RESOURCE_DESC textureDesc = {};
        textureDesc.Dimension        = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        textureDesc.Width            = metadata.width;
        textureDesc.Height           = metadata.height;
        textureDesc.DepthOrArraySize = metadata.arraySize;
        textureDesc.MipLevels        = metadata.mipLevels;
        textureDesc.Format           = metadata.format;
        textureDesc.SampleDesc       = { 1, 0 };
        textureDesc.Flags            = D3D12_RESOURCE_FLAG_NONE;

        ThrowIfFailed(device->CreateCommittedResource(
            &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
            D3D12_HEAP_FLAG_NONE,
            &textureDesc,
            D3D12_RESOURCE_STATE_COPY_DEST,
            nullptr,
            IID_PPV_ARGS(&m_Texture)));
    }

    // Create the intermediate upload buffer
    {
        const UINT64 uploadBufferSize = GetRequiredIntermediateSize(m_Texture.Get(), 0, metadata.arraySize * metadata.mipLevels);

        // Allocate memory space on the upload heap
        ThrowIfFailed(device->CreateCommittedResource(
            &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD),
            D3D12_HEAP_FLAG_NONE,
            &CD3DX12_RESOURCE_DESC::Buffer(uploadBufferSize),
            D3D12_RESOURCE_STATE_GENERIC_READ,
            nullptr,
            IID_PPV_ARGS(&m_TextureBuffer)));
    }
    
    // Upload pixels
    {
        const DirectX::Image* img = scratchImage.GetImages();
        D3D12_SUBRESOURCE_DATA textureData = {};
        textureData.pData      = img->pixels;
        textureData.RowPitch   = img->rowPitch;
        textureData.SlicePitch = img->slicePitch;

        UpdateSubresources(commandList.Get(), m_Texture.Get(), m_TextureBuffer.Get(), 0, 0, 1, &textureData);
    }

    // Transition to shader-readable state
    commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(m_Texture.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE));

    // Write SRV desc into slot 1 of the heap
    {
        UINT srvDescriptorSize = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        CD3DX12_CPU_DESCRIPTOR_HANDLE srvHandle(
            m_CBV_SRV_Heap->GetCPUDescriptorHandleForHeapStart(), 1, srvDescriptorSize);

        D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
        srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srvDesc.Format                  = metadata.format; // also textureDesc.format
        srvDesc.ViewDimension           = D3D12_SRV_DIMENSION_TEXTURE2D;
        srvDesc.Texture2D.MipLevels     = metadata.mipLevels;
        device->CreateShaderResourceView(m_Texture.Get(), &srvDesc, srvHandle);
    }
}

void Tutorial2::LoadCubemapTextureFromFile(
    const std::wstring& fileName,
    Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList2> commandList)
{
    auto device = Application::Get().GetDevice();

    fs::path filePath(fileName);
    if (!fs::exists(filePath)) throw std::exception("File not found.");

    TexMetadata metadata;
    ScratchImage scratchImage;
    ThrowIfFailed(LoadFromDDSFile(fileName.c_str(), DDS_FLAGS_FORCE_RGB, &metadata, scratchImage));

    if (metadata.dimension != TEX_DIMENSION_TEXTURE2D || !metadata.IsCubemap())
		throw std::exception("Only cubemap DDS textures are supported in the skybox.");

    // Create the final GPU texture resource
    {
        D3D12_RESOURCE_DESC textureDesc = {};
        textureDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        textureDesc.Width = metadata.width;
        textureDesc.Height = metadata.height;
        textureDesc.DepthOrArraySize = metadata.arraySize;
        textureDesc.MipLevels = static_cast<UINT16>(metadata.mipLevels);
        textureDesc.Format = metadata.format;
        textureDesc.SampleDesc = { 1, 0 };
        textureDesc.Flags = D3D12_RESOURCE_FLAG_NONE;

        ThrowIfFailed(device->CreateCommittedResource(
            &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
            D3D12_HEAP_FLAG_NONE,
            &textureDesc,
            D3D12_RESOURCE_STATE_COPY_DEST,
            nullptr,
            IID_PPV_ARGS(&m_SkyboxTexture)));
    }

    // Create the intermediate upload buffer
    {
        const UINT64 uploadBufferSize = GetRequiredIntermediateSize(m_SkyboxTexture.Get(), 0, metadata.arraySize * metadata.mipLevels);

        // Allocate memory space on the upload heap
        ThrowIfFailed(device->CreateCommittedResource(
            &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD),
            D3D12_HEAP_FLAG_NONE,
            &CD3DX12_RESOURCE_DESC::Buffer(uploadBufferSize),
            D3D12_RESOURCE_STATE_GENERIC_READ,
            nullptr,
            IID_PPV_ARGS(&m_SkyboxTextureBuffer)));
    }

    // upload all 6 faces
    {
        std::vector<D3D12_SUBRESOURCE_DATA> subresources;
		for (size_t face = 0; face < metadata.arraySize; ++face)
        {
            for (size_t mip = 0; mip < metadata.mipLevels; ++mip)
            {
                const DirectX::Image* img = scratchImage.GetImage(mip, face, 0);
                D3D12_SUBRESOURCE_DATA textureData = {};
                textureData.pData      = img->pixels;
                textureData.RowPitch   = img->rowPitch;
                textureData.SlicePitch = img->slicePitch;

                subresources.push_back(textureData);
            }
        }

		UpdateSubresources(commandList.Get(), m_SkyboxTexture.Get(), m_SkyboxTextureBuffer.Get(), 0, 0, static_cast<UINT>(subresources.size()), subresources.data());
    }

    // Transition to shader-readable state
    commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(m_SkyboxTexture.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE));

    // Write SRV desc into slot 2 of the heap
    {
        UINT srvDescriptorSize = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        CD3DX12_CPU_DESCRIPTOR_HANDLE srvHandle(
            m_CBV_SRV_Heap->GetCPUDescriptorHandleForHeapStart(), 2, srvDescriptorSize);

        D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
        srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srvDesc.Format                  = metadata.format;
        srvDesc.ViewDimension           = D3D12_SRV_DIMENSION_TEXTURECUBE;
        srvDesc.Texture2D.MipLevels     = static_cast<UINT>(metadata.mipLevels);
        device->CreateShaderResourceView(m_SkyboxTexture.Get(), &srvDesc, srvHandle);
    }
}

// ============================================================================
// Load content
// 1. Geometry buffers (vertex, index)
// 2. Descriptor heaps
// 3. Constant buffer + CBV
// 4. Texture + SRV
// 5. Root signature : skybox and main
// 6. PSO
// ============================================================================

bool Tutorial2::LoadContent()
{
    auto device = Application::Get().GetDevice();
    auto commandQueue = Application::Get().GetCommandQueue(D3D12_COMMAND_LIST_TYPE_DIRECT);
    auto commandList = commandQueue->GetCommandList();

    // 1. Geometry buffers (vertex, index)
    if (!LoadGLTF("Resources/shiba/scene.gltf"))
        return false;
    LoadGLTFMesh();

    // Upload vertex buffer data.
    ComPtr<ID3D12Resource> intermediateVertexBuffer;
    UpdateBufferResource(commandList,
        &m_VertexBuffer, &intermediateVertexBuffer,
        g_Vertices.size(), sizeof(VertexPosNormTex), g_Vertices.data());

    // Create the vertex buffer view.
    m_VertexBufferView.BufferLocation = m_VertexBuffer->GetGPUVirtualAddress();
    m_VertexBufferView.SizeInBytes    = g_Vertices.size() * sizeof(VertexPosNormTex);
    m_VertexBufferView.StrideInBytes  = sizeof(VertexPosNormTex);

    // Upload index buffer data.
    ComPtr<ID3D12Resource> intermediateIndexBuffer;
    UpdateBufferResource(commandList,
        &m_IndexBuffer, &intermediateIndexBuffer,
        g_Indices.size(), sizeof(uint32_t), g_Indices.data());

    // Create index buffer view.
    m_IndexBufferView.BufferLocation = m_IndexBuffer->GetGPUVirtualAddress();
    m_IndexBufferView.Format         = DXGI_FORMAT_R32_UINT;
    m_IndexBufferView.SizeInBytes    = static_cast<UINT>(g_Indices.size() * sizeof(uint32_t));

    // 1 bis. Geometry buffers (Skybox)
	ComPtr<ID3D12Resource> intermediateSkyboxVertexBuffer;
    UpdateBufferResource(commandList,
        &m_SkyboxVertexBuffer, &intermediateSkyboxVertexBuffer,
        _countof(Skybox::g_SkyboxVertices), sizeof(VertexPosNormTex), Skybox::g_SkyboxVertices);

	m_SkyboxVertexBufferView.BufferLocation = m_SkyboxVertexBuffer->GetGPUVirtualAddress();
    m_SkyboxVertexBufferView.SizeInBytes    = sizeof(Skybox::g_SkyboxVertices);
    m_SkyboxVertexBufferView.StrideInBytes  = sizeof(Skybox::SkyboxVertex);

	ComPtr<ID3D12Resource> intermediateSkyboxIndexBuffer;
    UpdateBufferResource(commandList,
		&m_SkyboxIndexBuffer, &intermediateSkyboxIndexBuffer,
        _countof(Skybox::g_SkyboxIndices), sizeof(uint16_t), Skybox::g_SkyboxIndices);

	m_SkyboxIndexBufferView.BufferLocation = m_SkyboxIndexBuffer->GetGPUVirtualAddress();
	m_SkyboxIndexBufferView.Format         = DXGI_FORMAT_R16_UINT;
	m_SkyboxIndexBufferView.SizeInBytes    = sizeof(Skybox::g_SkyboxIndices);

    // 2. Descriptor heaps
    // Create the descriptor heap for the depth-stencil view.
    D3D12_DESCRIPTOR_HEAP_DESC dsvHeapDesc = {};
    dsvHeapDesc.NumDescriptors = 1;
    dsvHeapDesc.Type           = D3D12_DESCRIPTOR_HEAP_TYPE_DSV;
    dsvHeapDesc.Flags          = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
    ThrowIfFailed(device->CreateDescriptorHeap(&dsvHeapDesc, IID_PPV_ARGS(&m_DSVHeap)));

	// Create the constant buffer view (CBV) and shader resource view (SRV) heap, respectively for the matrices and the texture.
    D3D12_DESCRIPTOR_HEAP_DESC heapDesc = {};
    heapDesc.NumDescriptors = 3; // slot 0 = CBV, slot 1 = model SRV, slot 2 = skybox SRV
    heapDesc.Flags          = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    heapDesc.Type           = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    ThrowIfFailed(device->CreateDescriptorHeap(&heapDesc, IID_PPV_ARGS(&m_CBV_SRV_Heap)));

    // 3. Constant buffer + CBV
    size_t totalSize = sizeof(Mat) + sizeof(CameraPositionData);
	const UINT constantBufferSize = static_cast<UINT>((totalSize + 255) & ~255u); // align to 256 bytes

    // Allocate memory space on the upload heap
    ThrowIfFailed(device->CreateCommittedResource(
        &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD),
        D3D12_HEAP_FLAG_NONE,
        &CD3DX12_RESOURCE_DESC::Buffer(constantBufferSize),
        D3D12_RESOURCE_STATE_GENERIC_READ,
        nullptr,
        IID_PPV_ARGS(&m_ConstantBuffer)));

    // Describe and create the constant buffer view (CBV).
    D3D12_CONSTANT_BUFFER_VIEW_DESC cbvDesc = {};
    cbvDesc.BufferLocation = m_ConstantBuffer->GetGPUVirtualAddress();
    cbvDesc.SizeInBytes    = constantBufferSize;
    device->CreateConstantBufferView(&cbvDesc, m_CBV_SRV_Heap->GetCPUDescriptorHandleForHeapStart());

    // Map the constant buffer to the virutal address space of the app to be able to initialize it using CPU memory map
	CD3DX12_RANGE readRange(0, 0); // We do not intend to read from this resource on the CPU.
	ThrowIfFailed(m_ConstantBuffer->Map(0, &readRange, reinterpret_cast<void**>(&m_pCBVDataBegin)));
    memcpy(m_pCBVDataBegin, &m_Matrices, sizeof(m_Matrices));
	memcpy(m_pCBVDataBegin + sizeof(Mat), &m_CameraPositionData, sizeof(CameraPositionData));

    // 4. Texture + SRV
    LoadTextureFromFile(L"Resources/shiba/textures/default_baseColor.png", commandList);
	LoadCubemapTextureFromFile(L"Resources/skybox/plains_sunset_4k.dds", commandList);

    // 5. Root signature
    D3D12_FEATURE_DATA_ROOT_SIGNATURE featureData = {};
    featureData.HighestVersion = D3D_ROOT_SIGNATURE_VERSION_1_1;
    if (FAILED(device->CheckFeatureSupport(D3D12_FEATURE_ROOT_SIGNATURE, &featureData, sizeof(featureData))))
        featureData.HighestVersion = D3D_ROOT_SIGNATURE_VERSION_1_0;

    CD3DX12_DESCRIPTOR_RANGE1 ranges[2];
    ranges[0].Init(D3D12_DESCRIPTOR_RANGE_TYPE_CBV, 1, 0, 0, D3D12_DESCRIPTOR_RANGE_FLAG_DATA_STATIC);
    ranges[1].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0, 0, D3D12_DESCRIPTOR_RANGE_FLAG_DATA_STATIC);

    // Allow input layout and deny unnecessary access to certain pipeline stages.
    D3D12_ROOT_SIGNATURE_FLAGS rootSignatureFlags =
        D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT |
        D3D12_ROOT_SIGNATURE_FLAG_DENY_HULL_SHADER_ROOT_ACCESS |
        D3D12_ROOT_SIGNATURE_FLAG_DENY_DOMAIN_SHADER_ROOT_ACCESS |
        D3D12_ROOT_SIGNATURE_FLAG_DENY_GEOMETRY_SHADER_ROOT_ACCESS;

    // ============ Skybox root signature ===============
    // @rootparam 0: descriptor table [CBV] (b0) : matrices
    // @rootparam 1: descriptor table [SRV] (t0) : cubemap
    CD3DX12_ROOT_PARAMETER1 skyboxRootParameters[2];
    skyboxRootParameters[0].InitAsDescriptorTable(1, &ranges[0], D3D12_SHADER_VISIBILITY_VERTEX);
    skyboxRootParameters[1].InitAsDescriptorTable(1, &ranges[1], D3D12_SHADER_VISIBILITY_PIXEL);

    D3D12_STATIC_SAMPLER_DESC skyboxSampler = {};
    skyboxSampler.Filter           = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
    skyboxSampler.AddressU         = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
    skyboxSampler.AddressV         = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
    skyboxSampler.AddressW         = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
    skyboxSampler.ComparisonFunc   = D3D12_COMPARISON_FUNC_NEVER;
    skyboxSampler.MaxLOD           = D3D12_FLOAT32_MAX;
    skyboxSampler.ShaderRegister   = 0;
    skyboxSampler.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;

    CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC skyboxRootSignatureDescription;
    skyboxRootSignatureDescription.Init_1_1(_countof(skyboxRootParameters), skyboxRootParameters, 1, &skyboxSampler, rootSignatureFlags);

    // Serialize the root signature.
    ComPtr<ID3DBlob> skyboxRootSignatureBlob;
    ComPtr<ID3DBlob> skyboxErrorBlob;
    ThrowIfFailed(D3DX12SerializeVersionedRootSignature(&skyboxRootSignatureDescription, featureData.HighestVersion, &skyboxRootSignatureBlob, &skyboxErrorBlob));

    // Create the root signature.
    ThrowIfFailed(device->CreateRootSignature(0, skyboxRootSignatureBlob->GetBufferPointer(), skyboxRootSignatureBlob->GetBufferSize(), IID_PPV_ARGS(&m_SkyboxRootSignature)));

    // ============ Main root signature ===============
    // @rootparam 0: descriptor table [CBV] (b0) : matrices + camera pos
    // @rootparam 1: descriptor table [SRV] (t0) : texture
    // @rootparam 2: constant               (b1) : directional light
    // @rootparam 3: constant               (b2) : material
    CD3DX12_ROOT_PARAMETER1 rootParameters[4];
    rootParameters[0].InitAsDescriptorTable(1, &ranges[0], D3D12_SHADER_VISIBILITY_ALL);
    rootParameters[1].InitAsDescriptorTable(1, &ranges[1], D3D12_SHADER_VISIBILITY_PIXEL);
    rootParameters[2].InitAsConstants(sizeof(DirectionalLight) / 4, 1, 0, D3D12_SHADER_VISIBILITY_PIXEL);
    rootParameters[3].InitAsConstants(sizeof(Material)         / 4, 2, 0, D3D12_SHADER_VISIBILITY_PIXEL);

    D3D12_STATIC_SAMPLER_DESC sampler = {};
    sampler.Filter           = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
    sampler.AddressU         = D3D12_TEXTURE_ADDRESS_MODE_BORDER;
    sampler.AddressV         = D3D12_TEXTURE_ADDRESS_MODE_BORDER;
    sampler.AddressW         = D3D12_TEXTURE_ADDRESS_MODE_BORDER;
    sampler.ComparisonFunc   = D3D12_COMPARISON_FUNC_NEVER;
    sampler.BorderColor      = D3D12_STATIC_BORDER_COLOR_TRANSPARENT_BLACK;
    sampler.MaxLOD           = D3D12_FLOAT32_MAX;
    sampler.ShaderRegister   = 0;
    sampler.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;

    CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC rootSignatureDescription;
    rootSignatureDescription.Init_1_1(_countof(rootParameters), rootParameters, 1, &sampler, rootSignatureFlags);

    // Serialize the root signature.
    ComPtr<ID3DBlob> rootSignatureBlob;
    ComPtr<ID3DBlob> errorBlob;
    ThrowIfFailed(D3DX12SerializeVersionedRootSignature(&rootSignatureDescription, featureData.HighestVersion, &rootSignatureBlob, &errorBlob));
        
    // Create the root signature.
    ThrowIfFailed(device->CreateRootSignature(0, rootSignatureBlob->GetBufferPointer(), rootSignatureBlob->GetBufferSize(), IID_PPV_ARGS(&m_RootSignature)));

    // 6. PSO
    // ============ Skybox PSO ===============
    // Load the vertex shader.
    ComPtr<ID3DBlob> skyboxVSBlob;
    ThrowIfFailed(D3DReadFileToBlob(L"SkyboxVS.cso", &skyboxVSBlob));

    // Load the pixel shader.
    ComPtr<ID3DBlob> skyboxPSBlob;
    ThrowIfFailed(D3DReadFileToBlob(L"SkyboxPS.cso", &skyboxPSBlob));

    // Create the vertex input layout
    D3D12_INPUT_ELEMENT_DESC skyboxInputLayout[] = {
        { "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, D3D12_APPEND_ALIGNED_ELEMENT, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
    };

	// Depth-stencil state: enable depth test but disable depth write
	CD3DX12_DEPTH_STENCIL_DESC depthStencilDesc(D3D12_DEFAULT);
    depthStencilDesc.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ZERO; // no writing
	depthStencilDesc.DepthFunc      = D3D12_COMPARISON_FUNC_LESS_EQUAL; // the skybox is rendered at the far plane (depth = 1.0)
    
	CD3DX12_RASTERIZER_DESC rasterizerDesc(D3D12_DEFAULT);
	rasterizerDesc.CullMode = D3D12_CULL_MODE_FRONT; // the camera is inside the cube

    struct SkyboxPipelineStateStream
    {
        CD3DX12_PIPELINE_STATE_STREAM_ROOT_SIGNATURE        pRootSignature;
        CD3DX12_PIPELINE_STATE_STREAM_INPUT_LAYOUT          InputLayout;
        CD3DX12_PIPELINE_STATE_STREAM_PRIMITIVE_TOPOLOGY    PrimitiveTopologyType;
        CD3DX12_PIPELINE_STATE_STREAM_VS                    VS;
        CD3DX12_PIPELINE_STATE_STREAM_PS                    PS;
        CD3DX12_PIPELINE_STATE_STREAM_DEPTH_STENCIL         DS;
        CD3DX12_PIPELINE_STATE_STREAM_DEPTH_STENCIL_FORMAT  DSVFormat;
        CD3DX12_PIPELINE_STATE_STREAM_RASTERIZER            Rasterizer;
        CD3DX12_PIPELINE_STATE_STREAM_RENDER_TARGET_FORMATS RTVFormats;
    } skyboxPSS;

    D3D12_RT_FORMAT_ARRAY skyboxRtvFormats = {};
    skyboxRtvFormats.NumRenderTargets = 1;
    skyboxRtvFormats.RTFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;

    skyboxPSS.pRootSignature        = m_SkyboxRootSignature.Get();
    skyboxPSS.InputLayout           = { skyboxInputLayout, _countof(skyboxInputLayout) };
    skyboxPSS.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    skyboxPSS.VS                    = CD3DX12_SHADER_BYTECODE(skyboxVSBlob.Get());
    skyboxPSS.PS                    = CD3DX12_SHADER_BYTECODE(skyboxPSBlob.Get());
	skyboxPSS.DS                    = depthStencilDesc;
    skyboxPSS.DSVFormat             = DXGI_FORMAT_D32_FLOAT;
	skyboxPSS.Rasterizer            = rasterizerDesc;
    skyboxPSS.RTVFormats            = skyboxRtvFormats;

    D3D12_PIPELINE_STATE_STREAM_DESC skyboxPSSDesc = { sizeof(SkyboxPipelineStateStream), &skyboxPSS };
    ThrowIfFailed(device->CreatePipelineState(&skyboxPSSDesc, IID_PPV_ARGS(&m_SkyboxPipelineState)));

    // ============ Main PSO ===============
    // Load the vertex shader.
    ComPtr<ID3DBlob> vertexShaderBlob;
    ThrowIfFailed(D3DReadFileToBlob(L"VertexShader.cso", &vertexShaderBlob));

    // Load the pixel shader.
    ComPtr<ID3DBlob> pixelShaderBlob;
    ThrowIfFailed(D3DReadFileToBlob(L"PixelShader.cso", &pixelShaderBlob));

    // Create the vertex input layout
    D3D12_INPUT_ELEMENT_DESC inputLayout[] = {
        { "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, D3D12_APPEND_ALIGNED_ELEMENT, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
        { "NORMAL",   0, DXGI_FORMAT_R32G32B32_FLOAT, 0, D3D12_APPEND_ALIGNED_ELEMENT, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
        { "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT,    0, D3D12_APPEND_ALIGNED_ELEMENT, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
    };

    struct PipelineStateStream
    {
        CD3DX12_PIPELINE_STATE_STREAM_ROOT_SIGNATURE        pRootSignature;
        CD3DX12_PIPELINE_STATE_STREAM_INPUT_LAYOUT          InputLayout;
        CD3DX12_PIPELINE_STATE_STREAM_PRIMITIVE_TOPOLOGY    PrimitiveTopologyType;
        CD3DX12_PIPELINE_STATE_STREAM_VS                    VS;
        CD3DX12_PIPELINE_STATE_STREAM_PS                    PS;
        CD3DX12_PIPELINE_STATE_STREAM_DEPTH_STENCIL_FORMAT  DSVFormat;
        CD3DX12_PIPELINE_STATE_STREAM_RENDER_TARGET_FORMATS RTVFormats;
    } pipelineStateStream;

    D3D12_RT_FORMAT_ARRAY rtvFormats = {};
    rtvFormats.NumRenderTargets = 1;
    rtvFormats.RTFormats[0]     = DXGI_FORMAT_R8G8B8A8_UNORM;

    pipelineStateStream.pRootSignature        = m_RootSignature.Get();
    pipelineStateStream.InputLayout           = { inputLayout, _countof(inputLayout) };
    pipelineStateStream.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    pipelineStateStream.VS                    = CD3DX12_SHADER_BYTECODE(vertexShaderBlob.Get());
    pipelineStateStream.PS                    = CD3DX12_SHADER_BYTECODE(pixelShaderBlob.Get());
    pipelineStateStream.DSVFormat             = DXGI_FORMAT_D32_FLOAT;
    pipelineStateStream.RTVFormats            = rtvFormats;

    D3D12_PIPELINE_STATE_STREAM_DESC pipelineStateStreamDesc = { sizeof(PipelineStateStream), &pipelineStateStream };
    ThrowIfFailed(device->CreatePipelineState(&pipelineStateStreamDesc, IID_PPV_ARGS(&m_PipelineState)));

    auto fenceValue = commandQueue->ExecuteCommandList(commandList);
    commandQueue->WaitForFenceValue(fenceValue);

    m_ContentLoaded = true;

    // Resize/Create the depth buffer.
    ResizeDepthBuffer(GetClientWidth(), GetClientHeight());

    return true;
}

void Tutorial2::UnloadContent()
{
    m_ContentLoaded = false;
}

// ============================================================================
// Resize functions
// ============================================================================

void Tutorial2::ResizeDepthBuffer(int width, int height)
{
    if (!m_ContentLoaded) return;

    // Flush any GPU commands that might be referencing the depth buffer.
    Application::Get().Flush();

    width = std::max(1, width);
    height = std::max(1, height);

    auto device = Application::Get().GetDevice();

    // Resize screen dependent resources.
    // Create a depth buffer.
    D3D12_CLEAR_VALUE optimizedClearValue = {};
    optimizedClearValue.Format            = DXGI_FORMAT_D32_FLOAT;
    optimizedClearValue.DepthStencil      = { 1.0f, 0 };

    ThrowIfFailed(device->CreateCommittedResource(
        &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
        D3D12_HEAP_FLAG_NONE,
        &CD3DX12_RESOURCE_DESC::Tex2D(DXGI_FORMAT_D32_FLOAT, width, height,
            1, 0, 1, 0, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL),
        D3D12_RESOURCE_STATE_DEPTH_WRITE,
        &optimizedClearValue,
        IID_PPV_ARGS(&m_DepthBuffer)
    ));

    // Update the depth-stencil view.
    D3D12_DEPTH_STENCIL_VIEW_DESC dsv = {};
    dsv.Format                        = DXGI_FORMAT_D32_FLOAT;
    dsv.ViewDimension                 = D3D12_DSV_DIMENSION_TEXTURE2D;
    dsv.Texture2D.MipSlice            = 0;
    dsv.Flags                         = D3D12_DSV_FLAG_NONE;

    device->CreateDepthStencilView(m_DepthBuffer.Get(), &dsv,
        m_DSVHeap->GetCPUDescriptorHandleForHeapStart());
}

void Tutorial2::OnResize(ResizeEventArgs& e)
{
    if (e.Width != GetClientWidth() || e.Height != GetClientHeight())
    {
        super::OnResize(e);
        float aspectRatio = e.Width / (float)e.Height;
        m_Camera.set_Projection(45.0f, aspectRatio, 0.1f, 100.0f);
        m_Viewport = CD3DX12_VIEWPORT(0.0f, 0.0f,static_cast<float>(e.Width), static_cast<float>(e.Height));
        ResizeDepthBuffer(e.Width, e.Height);
    }
}

// ============================================================================
// Update
// ============================================================================

void Tutorial2::OnUpdate(UpdateEventArgs& e)
{
    static uint64_t frameCount = 0;
    static double totalTime = 0.0;

    super::OnUpdate(e);

    totalTime += e.ElapsedTime;
    frameCount++;
    if (totalTime > 1.0)
    {
        double fps = frameCount / totalTime;

        char buffer[512];
        sprintf_s(buffer, "FPS: %f\n", fps);
        OutputDebugStringA(buffer);

        frameCount = 0;
        totalTime = 0.0;
    }

    // Update the camera.
    float speed = 4.0f * static_cast<float>(e.ElapsedTime);
    XMVECTOR cameraTranslate = XMVectorSet(m_Right - m_Left, 0.0f, m_Forward - m_Backward, 1.0f) * speed;
    XMVECTOR cameraPan = XMVectorSet(0.0f, m_Up - m_Down, 0.0f, 1.0f) * speed;
    m_Camera.Translate(cameraTranslate, Space::Local);
    m_Camera.Translate(cameraPan, Space::Local);
    XMVECTOR cameraRotation = XMQuaternionRotationRollPitchYaw(XMConvertToRadians(m_Pitch), XMConvertToRadians(m_Yaw), 0.0f);
    m_Camera.set_Rotation(cameraRotation);

    OnRender();
}

// ============================================================================
// Helper functions
// ============================================================================

// Transition a resource
void Tutorial2::TransitionResource(
    Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList2> commandList,
    Microsoft::WRL::ComPtr<ID3D12Resource> resource,
    D3D12_RESOURCE_STATES beforeState, D3D12_RESOURCE_STATES afterState)
{
    CD3DX12_RESOURCE_BARRIER barrier = CD3DX12_RESOURCE_BARRIER::Transition(resource.Get(), beforeState, afterState);
    commandList->ResourceBarrier(1, &barrier);
}

// Clear a render target.
void Tutorial2::ClearRTV(Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList2> commandList, D3D12_CPU_DESCRIPTOR_HANDLE rtv, FLOAT* clearColor)
{
    commandList->ClearRenderTargetView(rtv, clearColor, 0, nullptr);
}

void Tutorial2::ClearDepth(Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList2> commandList, D3D12_CPU_DESCRIPTOR_HANDLE dsv, FLOAT depth)
{
    commandList->ClearDepthStencilView(dsv, D3D12_CLEAR_FLAG_DEPTH, depth, 0, 0, nullptr);
}

// ============================================================================
// Per-frame matrix upload
// ============================================================================

static void XM_CALLCONV ComputeMatrices(FXMMATRIX model, CXMMATRIX view, CXMMATRIX projection, Mat& mat)
{
	CXMMATRIX viewProjection = XMMatrixMultiply(view, projection);
    mat.ModelMatrix                 = model;
    mat.ModelViewMatrix             = model * view;
    mat.InverseTransposeModelMatrix = XMMatrixTranspose(XMMatrixInverse(nullptr, mat.ModelMatrix));
    mat.ModelViewProjectionMatrix   = model * viewProjection;

	mat.ViewMatrix                  = view;
	mat.ProjectionMatrix            = projection;
}

void Tutorial2::ComputeAndUploadModelMatrices(Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList2> commandList)
{
    // glTF use a right-handed coordinate system, the model is rotated to be displayed correctly in DX3D12 left-handed coordinate system.
    XMMATRIX rotationMatrix       = XMMatrixRotationX(XMConvertToRadians(-90.0f))
                                  * XMMatrixRotationY(XMConvertToRadians(180.0f));
    XMMATRIX translationMatrix    = XMMatrixIdentity();
    XMMATRIX scaleMatrix          = XMMatrixIdentity();
    XMMATRIX worldMatrix          = scaleMatrix * rotationMatrix * translationMatrix;
    XMMATRIX viewMatrix           = m_Camera.get_ViewMatrix();
	XMMATRIX projectionMatrix     = m_Camera.get_ProjectionMatrix();

    // Update the m_Matrices constant buffer
    ComputeMatrices(worldMatrix, viewMatrix, projectionMatrix, m_Matrices);
    // Copy constant buffer data on the upload heap
	memcpy(m_pCBVDataBegin, &m_Matrices, sizeof(m_Matrices));
}

// ============================================================================
// Render
// ============================================================================

void Tutorial2::OnRender()
{
    super::OnRender();

    auto commandQueue = Application::Get().GetCommandQueue(D3D12_COMMAND_LIST_TYPE_DIRECT);
    auto commandList  = commandQueue->GetCommandList();

    UINT currentBackBufferIndex = m_pWindow->GetCurrentBackBufferIndex();
    auto backBuffer             = m_pWindow->GetCurrentBackBuffer();
    auto rtv                    = m_pWindow->GetCurrentRenderTargetView();
    auto dsv                    = m_DSVHeap->GetCPUDescriptorHandleForHeapStart();

    // Clear the render targets.
    {
        TransitionResource(commandList, backBuffer, D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_RENDER_TARGET);
        FLOAT clearColor[] = { 0.4f, 0.6f, 0.9f, 1.0f };
        ClearRTV(commandList, rtv, clearColor);
        ClearDepth(commandList, dsv);
    }

    // Update constant buffer with current frame data
    ComputeAndUploadModelMatrices(commandList);
    m_CameraPositionData.CameraPos = m_Camera.get_Translation();
    memcpy(m_pCBVDataBegin + sizeof(Mat), &m_CameraPositionData, sizeof(CameraPositionData));

    // ======== Draw skybox =========
    {
        // Pipeline setup
        commandList->SetPipelineState(m_SkyboxPipelineState.Get());
        commandList->SetGraphicsRootSignature(m_SkyboxRootSignature.Get());

        // Bind descriptor heap
        ID3D12DescriptorHeap* descHeaps[] = { m_CBV_SRV_Heap.Get() };
        commandList->SetDescriptorHeaps(_countof(descHeaps), descHeaps);

        // @rootparam 0: CBV (slot 0 of heap)
        commandList->SetGraphicsRootDescriptorTable(0, m_CBV_SRV_Heap->GetGPUDescriptorHandleForHeapStart());

        // @rootparam 1: cubemap SRV (slot 2 of heap)
        UINT srvDescriptorSize = Application::Get().GetDevice()->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        CD3DX12_GPU_DESCRIPTOR_HANDLE skyboxGpuHandle(m_CBV_SRV_Heap->GetGPUDescriptorHandleForHeapStart(), 2, srvDescriptorSize);
        commandList->SetGraphicsRootDescriptorTable(1, skyboxGpuHandle);

        commandList->RSSetViewports(1, &m_Viewport);
        commandList->RSSetScissorRects(1, &m_ScissorRect);
        commandList->OMSetRenderTargets(1, &rtv, FALSE, &dsv);
        commandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        commandList->IASetVertexBuffers(0, 1, &m_SkyboxVertexBufferView);
        commandList->IASetIndexBuffer(&m_SkyboxIndexBufferView);

        commandList->DrawIndexedInstanced(_countof(Skybox::g_SkyboxIndices), 1, 0, 0, 0);
    }

    // ======== Draw model ==========
    {
        // Pipeline setup
        commandList->SetPipelineState(m_PipelineState.Get());
        commandList->SetGraphicsRootSignature(m_RootSignature.Get());

        // Bind descriptor heap
        ID3D12DescriptorHeap* descHeaps[] = { m_CBV_SRV_Heap.Get() };
        commandList->SetDescriptorHeaps(_countof(descHeaps), descHeaps);

        // @rootparam 0: CBV (slot 0 of heap)
        commandList->SetGraphicsRootDescriptorTable(0, m_CBV_SRV_Heap->GetGPUDescriptorHandleForHeapStart());

        // @rootparam 1: SRV texture (slot 1 of heap)
        UINT srvDescriptorSize = Application::Get().GetDevice()->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        CD3DX12_GPU_DESCRIPTOR_HANDLE srvGpuHandle(m_CBV_SRV_Heap->GetGPUDescriptorHandleForHeapStart(), 1, srvDescriptorSize);
        commandList->SetGraphicsRootDescriptorTable(1, srvGpuHandle);

        // @rootparam 2 & 3: inline constants
        commandList->SetGraphicsRoot32BitConstants(2, sizeof(DirectionalLight) / 4, &g_DirLight, 0);
        commandList->SetGraphicsRoot32BitConstants(3, sizeof(Material) / 4, &g_ModelMat, 0);

        commandList->RSSetViewports(1, &m_Viewport);
        commandList->RSSetScissorRects(1, &m_ScissorRect);
        commandList->OMSetRenderTargets(1, &rtv, FALSE, &dsv);
        commandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        commandList->IASetVertexBuffers(0, 1, &m_VertexBufferView);
        commandList->IASetIndexBuffer(&m_IndexBufferView);

        commandList->DrawIndexedInstanced(g_Indices.size(), 1, 0, 0, 0);
    }

    // Present
    {
        TransitionResource(commandList, backBuffer, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PRESENT);
        m_FenceValues[currentBackBufferIndex] = commandQueue->ExecuteCommandList(commandList);
        currentBackBufferIndex = m_pWindow->Present();
        commandQueue->WaitForFenceValue(m_FenceValues[currentBackBufferIndex]);
    }
}

// ============================================================================
// Input handling
// ============================================================================

void Tutorial2::OnKeyPressed(KeyEventArgs& e)
{
    super::OnKeyPressed(e);

    switch (e.Key)
    {
    case KeyCode::Escape: Application::Get().Quit(0); break;
    case KeyCode::Enter:
        if (e.Alt)
        {
    case KeyCode::F11: m_pWindow->ToggleFullscreen(); break;
        }
    case KeyCode::V: m_pWindow->ToggleVSync(); break;

    // Move in the scene
    case KeyCode::R:
        // Reset camera transform
        m_Camera.set_Translation(m_pAlignedCameraData->m_InitialCamPos);
        m_Camera.set_Rotation(m_pAlignedCameraData->m_InitialCamRot);
        m_Pitch = 0.0f;
        m_Yaw = 0.0f;
        break;
    
    case KeyCode::Up: [[fallthrough]];
    case KeyCode::W: m_Forward = 1.0f; break;

    case KeyCode::Left: [[fallthrough]];
    case KeyCode::A: m_Left = 1.0f; break;

    case KeyCode::Down: [[fallthrough]];
    case KeyCode::S: m_Backward = 1.0f; break;

    case KeyCode::Right: [[fallthrough]];
    case KeyCode::D: m_Right = 1.0f; break;

    case KeyCode::Q: m_Down = 1.0f; break;
    case KeyCode::E: m_Up = 1.0f; break;
    }
}

void Tutorial2::OnKeyReleased(KeyEventArgs& e)
{
    switch (e.Key)
    {
    case KeyCode::Up: [[fallthrough]];
    case KeyCode::W: m_Forward = 0.0f; break;

    case KeyCode::Left: [[fallthrough]];
    case KeyCode::A: m_Left = 0.0f; break;

    case KeyCode::Down: [[fallthrough]];
    case KeyCode::S: m_Backward = 0.0f; break;

    case KeyCode::Right: [[fallthrough]];
    case KeyCode::D: m_Right = 0.0f; break;

    case KeyCode::Q: m_Down = 0.0f; break;
    case KeyCode::E: m_Up = 0.0f; break;
    }
}

void Tutorial2::OnMouseMoved(MouseMotionEventArgs& e)
{
    const float mouseSpeed = 0.1f;

    e.RelX = e.X - m_PreviousMouseX;
    e.RelY = e.Y - m_PreviousMouseY;

    m_PreviousMouseX = e.X;
    m_PreviousMouseY = e.Y;
    
    if (e.LeftButton)
    {
        m_Pitch -= e.RelY * mouseSpeed;
        m_Pitch = std::clamp(m_Pitch, -90.0f, 90.0f);

        m_Yaw -= e.RelX * mouseSpeed;
    }
}

void Tutorial2::OnMouseWheel(MouseWheelEventArgs& e)
{
    float fov = std::clamp(m_Camera.get_FoV() - e.WheelDelta, 12.0f, 90.0f);
    m_Camera.set_FoV(fov);
    char buffer[256];
    sprintf_s(buffer, "FoV: %f\n", fov);
    OutputDebugStringA(buffer);
}
