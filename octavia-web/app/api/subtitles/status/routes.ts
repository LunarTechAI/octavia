import { NextRequest, NextResponse } from 'next/server';
import { api } from '@/lib/api';

export async function GET(
  request: NextRequest,
  { params }: { params: { jobId: string } }
) {
  try {
    const searchParams = request.nextUrl.searchParams;
    const jobId = searchParams.get('jobId') || params.jobId;
    
    if (!jobId) {
      return NextResponse.json(
        { success: false, error: 'Job ID is required' },
        { status: 400 }
      );
    }
    
    
    return NextResponse.json({
      success: true,
      status: 'processing',
      progress: 65,
      estimated_time: '8 minutes'
    });
  } catch (error) {
    return NextResponse.json(
      { success: false, error: 'Failed to fetch status' },
      { status: 500 }
    );
  }
}