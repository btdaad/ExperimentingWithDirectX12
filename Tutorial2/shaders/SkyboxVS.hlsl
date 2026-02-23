cbuffer SceneCB : register(b0)
{
    matrix ModelMatrix;
    matrix ModelViewMatrix;
    matrix InverseTransposeModelMatrix;
    matrix ModelViewProjectionMatrix;

    float3 CameraPosition;
    float _pad1;
    
    matrix ViewMatrix;
    matrix ProjectionMatrix;
};

struct SkyboxVertex
{
    float3 Position : POSITION;
};

struct VertexShaderOutput
{
    float3 TexCoord : TEXCOORD;
    float4 Position : SV_POSITION;
};

VertexShaderOutput main(SkyboxVertex IN)
{
    VertexShaderOutput OUT;

    OUT.TexCoord = IN.Position;

    matrix viewNoTranslation = ViewMatrix;
    viewNoTranslation[3] = float4(0, 0, 0, 1); // we don't want the skybox to move with the camera, so we remove the translation component of the view matrix

    float4 viewPos = mul(viewNoTranslation, float4(IN.Position, 1.0));
    float4 clipPos = mul(ProjectionMatrix, viewPos);

    OUT.Position = clipPos.xyww; // force depth to 1.0

    return OUT;
}