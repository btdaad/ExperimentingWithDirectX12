cbuffer SceneCB : register(b0)
{
    matrix ModelMatrix;
    matrix ModelViewMatrix;
    matrix InverseTransposeModelMatrix;
    matrix ModelViewProjectionMatrix;

    float3 CameraPosition;
    float _pad1; // to align to 16 bytes
    
    matrix ViewMatrix;
    matrix ProjectionMatrix;
};

struct VertexPosNormTex
{
    float3 Position : POSITION;
    float3 Normal : NORMAL;
    float2 Uv : TEXCOORD;
};

struct VertexShaderOutput
{
    float4 PositionVS : POSITION;
    float4 NormalVS : NORMAL;
    float2 UvVS : TEXCOORD;
    float4 WorldPosVS : WORLD_POSITION;
    float4 Position : SV_Position;
};

VertexShaderOutput main(VertexPosNormTex IN)
{
    VertexShaderOutput OUT;

    OUT.Position = mul(ModelViewProjectionMatrix, float4(IN.Position, 1.0f));
    OUT.PositionVS = mul(ModelViewMatrix, float4(IN.Position, 1.0f));
    OUT.NormalVS = mul(InverseTransposeModelMatrix, float4(IN.Normal, 0.0f));
    OUT.UvVS = IN.Uv;
    OUT.WorldPosVS = mul(ModelMatrix, float4(IN.Position, 1.0f));
    
    return OUT;
}